import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.transforms import identity, orbit_sampling, AffineTransformation
import src.transforms as transforms
from sklearn.cluster import KMeans
from pylab import *
from tqdm import tqdm
import matplotlib as mpl
import torch.nn.functional as F
from src.measure import curvature, entropy
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap


def highlight_subplot(ax, color='red', linewidth=8):
    ax_coor = ax.axis()
    # xy, width, height
    rec = Rectangle((ax_coor[0], ax_coor[2]), ax_coor[1], 1,
                    fill=True, lw=linewidth, color=color)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)


class ITSScheduledInference:

    def __init__(self, model, transformation, n_samples=16, ncs_weight=0.5,
                 show_progress=True, labels=None, plot=False, domain=None, device='cuda'):
        self.model = model
        self.transformation = transformation
        self.n_samples = n_samples
        self.domain = domain
        self.show_progress = show_progress
        self.labels = labels
        self.plot = plot
        self.plot_memory = {"orbit": [], "embedding": [], "score": [], "n_max": [], "T": []}
        self.top_k = 1
        self.ncs_weight = ncs_weight
        self.batch_size = None
        self.input_shape = None
        self.true_label_idx = None
        self.device = device
        self.mc_steps = None
        self.transformation_schedule = transformation
        self.hypothesis_weight = None
        # batch x k x 3 x 3
        self.transformation_matrix = None

    def orbit_class_weight(self, z):
        n_classes = z.shape[-1]
        # Max likely class prediction
        z = z.argmax(dim=-1)
        # count number of class occurrences along the orbit
        counts = torch.stack([torch.bincount(z_part, minlength=n_classes) for z_part in z.flatten(end_dim=-2)])
        # batch x top_k x classes
        counts = counts.unflatten(0, (self.batch_size, self.top_k))
        return counts
        # normalize by orbit length
        # weight = counts / self.n_samples
        # weight -= weight.mean(dim=-1, keepdims=True)
        # batch x top_k x n_samples filled with normalized weights
        # return torch.gather(weight, dim=-1, index=z)

    def infer(self, x, top_k=1, true_label_idx=None, mc_steps=50, plot_idx=0, line_thickness=1, fontsize=12):
        """ inference
        :param x: (batch x ...)
        :return:
        """
        self.top_k = top_k
        self.mc_steps = mc_steps
        self.true_label_idx = true_label_idx
        self.batch_size = int(x.shape[0])
        self.input_shape = list(x.shape[1:])
        self.transformation_matrix = transforms.identity((self.batch_size, self.top_k)).to(self.device)
        self.plot_memory = {"orbit": [], "embedding": [], "score": [], "n_max": [], "T": []}
        # ensure consistent shapes
        x = torch.tile(x[:, None, ...], [1, self.top_k, 1, 1, 1])
        # outer loop: iterate over all possible transformations
        progress_bar = tqdm(range(len(self.transformation_schedule))) \
            if self.show_progress else range(len(self.transformation_schedule))
        for i in progress_bar:
            # keep the same x as input but the transformation matrix is composed during search
            x_trans, pred_class, pred_emb, scores = self.__breadth_first_search_step(x, level=i)
        if self.plot:
            self.__plot_tree(batch_index=plot_idx, line_thickness=line_thickness, fontsize=fontsize)
        # hypothesis testing: change of mind
        weight = torch.zeros([self.batch_size, self.top_k, pred_emb.shape[-1]])
        for i in range(len(self.transformation)):
            z = torch.tensor(self.plot_memory['embedding'][i])
            weight += self.orbit_class_weight(z)
        # get class of the hypothesis and count that
        weight = weight.to(x.device)
        weight = torch.gather(weight, dim=-1, index=pred_class[..., None]).squeeze(2)
        self.hypothesis_weight = weight
        # weight = weight.max(axis=-1).values
        indices = torch.sort(weight, dim=-1, descending=True).indices
        # indices = torch.sort(scores, dim=-1, descending=True).indices
        x_trans = torch.gather(x_trans, dim=1, index=torch.tile(
            indices[..., None, None, None], [1, 1] + list(x_trans.shape[-3:])))#.squeeze(2)
        pred_class = torch.gather(pred_class, dim=1, index=indices)
        pred_emb = torch.gather(pred_emb, dim=1, index=torch.tile(
            indices[..., None], [1, 1, pred_emb.shape[-1]]))
        return x_trans, pred_class, pred_emb

    def __evaluate(self, z):
        """ returns score (goal maximize)
        :param z: (batch x top_k x samples x classes)
        :return: (batch x top_k x samples)
        """
        # return torch.max(z[..., 1:self.n_samples+1, :], dim=-1).values
        energies = torch.log(torch.exp(z).sum(dim=-1))
        # as z can be negative, so the energy, so the curvature - any weighting can be misleading
        # energies += torch.min(energies, dim=-1, keepdim=True).values
        # weight = orbit_class_weight(z)
        # print(weight[23])
        # energies *= (10 * weight + 1)
        energies = gaussian_filter1d(energies.detach().cpu(), sigma=2, radius=3, mode='nearest') # 2,3
        confidence = -curvature(torch.tensor(energies, device=z.device))[..., 1:self.n_samples+1]

        return confidence
        # return -torch.tensor(energies[..., 1:self.n_samples+1], device=z.device)

    def __monte_carlo(self, x):
        """ Note that this adds stochasticity to the confidence.
        Therefore, the score of later levels can be lower than of early predictions.
        :param x:
        :return:
        """
        # x = x.type(torch.cuda.FloatTensor)
        # this ensures at deterministic predictions for mc = 1
        if self.mc_steps == 1:
            self.model.eval()
            return self.model(x)
        self.model.train()
        predictions = []
        for i in range(self.mc_steps):
            with torch.no_grad():
                predictions.append(self.model(x))
        return torch.stack(predictions).mean(dim=0)

    def __orbit_centroid_step(self, x, transformation, domain):
        """ todo remove pred class
        :param x: (batch x channel x height x width)
        :param transformations: (batch, )
        :param domain: (batch, )
        :return:
        """
        # batch * k x samples x 3 x 3
        T = self.transformation_matrix.flatten(end_dim=1)
        T = T[:, None].expand((-1, self.n_samples + 2, -1, -1))
        # (batch x samples x classes)
        orbit, T = orbit_sampling(x.flatten(end_dim=1), transformation,
                                  n_samples=self.n_samples, domain=[domain,], T=T)
        # resnet logits model.fn; vgg logits model.classifier[6]
        # torch default return un-normalized scores/logits
        z = self.__monte_carlo(orbit.flatten(end_dim=2))
        # remember extend of 1 for each side
        z = z.unflatten(0, (self.batch_size, self.top_k, self.n_samples + 2))
        c = torch.argmax(z[:, :, 1:self.n_samples+1], dim=-1)
        # squeeze away single orbit dim as len(T) = 1
        orbit = orbit.unflatten(0, (self.batch_size, self.top_k)).squeeze(dim=2)
        # T is a list of length 1
        T = T[0].unflatten(0, (self.batch_size, self.top_k))
        return orbit, z, c, T

    def __orbit_centroid_scores(self, x, level):
        """
        :param x:
        :param level:
        :return:
            - orbit: (B x K x O x Ch x H x W)
            - score: (B x K x O)
            - embedding: (B x K x O x C)
            - pred_class: (B x K x O)
            - T: (B x K x O x 3 x 3)
        """
        transformation = self.transformation_schedule[level]
        domain = self.domain[level]
        orbit, embedding, pred_class, transformation_matrix = self.__orbit_centroid_step(x, transformation, domain)
        score = self.__evaluate(embedding)
        return (orbit[:, :, 1:self.n_samples+1], score,
                embedding[:, :, 1:self.n_samples+1], pred_class,
                transformation_matrix[:, :, 1:self.n_samples+1])

    def select_candidates(self, score, pred_class, level):
        """ Selects candidates based on the score.
        For the first level only the k=0 is considered as all other k are identical anyway.
        Further, candidates of the same class are avoided.
        :param score: (B x K x O)
        :param pred_class: (B x K x O)
        :param level: int
        :return: (B x K)
        """
        if level == 0:
            score, pred_class = score[:, 0], pred_class[:, 0]
        n_max = torch.zeros((self.batch_size, self.top_k), device=score.device, dtype=torch.int64)
        for i in range(self.top_k):
            # find the first/next max score index
            if level == 0:
                # index: (B x 1 x 1)
                index = torch.argmax(score, dim=-1)
                # get the class associated with this index (on the orbit)
                c = pred_class[torch.arange(score.shape[0]), index]
                # replace every occurrence of the class with a zero score
                score = torch.where(pred_class == c[:, None], torch.full_like(score, -torch.inf), score)
                # update the top-k list
                n_max[:, i] = index
            else:
                # index: (B, )
                index = torch.argmax(score[:, i], dim=-1)
                # get the class associated with this index (on the orbit)
                c = torch.gather(pred_class[:, i], -1, index[:, None])[:, None]
                # replace every occurrence of the class with a zero score
                score = torch.where(pred_class == c, torch.full_like(score, -torch.inf), score)
                # update the top-k list
                n_max[:, i] = index
        return n_max

    def __breadth_first_search_step(self, x, level):
        """ Performs one breadth first search step.
        :param x: (batch x top_k x channel x height x width)
        :param transformations: open transformations
        :return: a tuple of
            - (incumbent) best performing input transformation,
            - (primal) highest scores to the centroids,
            - (transformation) the transformation achieved the former
            - (prediction) the model output
        """
        # compute the highest class scores and orbits
        orbit, score, embedding, pred_class, transformation_matrix = self.__orbit_centroid_scores(x, level)
        # select the k best samples from the orbit -> (batch x k)
        n_max = self.select_candidates(score, pred_class, level)
        # update incumbent; best_samples: (batch x top_k x samples x channel x height x width)
        # 128, 5, 16, 1, 28, 28
        best_x = torch.gather(orbit, dim=2, index=torch.tile(
            n_max[..., None, None, None, None], [1, 1, 1] + list(orbit.shape[-3:]))).squeeze(2)
        # 128, 5, 16, 3, 3
        self.transformation_matrix = torch.gather(transformation_matrix, dim=2, index=torch.tile(
            n_max[..., None, None, None], [1, 1, 1, 3, 3])).squeeze(2)
        # 128, 5, 16
        best_pred = torch.gather(pred_class, dim=2, index=torch.tile(
            n_max[..., None], [1, 1, 1])).squeeze(2)
        # 128, 5, 16, 9
        best_emb = torch.gather(embedding, dim=2, index=torch.tile(
            n_max[..., None, None], [1, 1, 1, embedding.shape[-1]])).squeeze(2)
        # plot the entire row/level
        # if self.plot:
        self.plot_memory["orbit"].append(orbit.detach().cpu().numpy())
        self.plot_memory["embedding"].append(embedding.detach().cpu().numpy())
        self.plot_memory["score"].append(score.detach().cpu().numpy())
        self.plot_memory["n_max"].append(n_max.detach().cpu().numpy())
        self.plot_memory["T"].append(transformation_matrix.detach().cpu().numpy())
        # todo is best_emb needed somewhere?
        best_scores = torch.gather(score, dim=2, index=n_max[..., None]).squeeze(2)
        return best_x, best_pred, best_emb, best_scores

    def __plot_tree(self, batch_index=0, line_thickness=1, fontsize=12):
        fig = plt.figure(layout='constrained', figsize=(self.n_samples * 1.5, self.top_k * 2 * len(self.transformation)))
        subfigs = fig.subfigures(len(self.transformation), 1, wspace=0.07)
        subfigs = subfigs if len(self.transformation) > 1 else [subfigs, ]

        # normalize scores globally over the entire tree
        # scores: level x (B x K x O)
        stacked_tensor = np.stack(self.plot_memory["score"]).transpose((1, 0, 2, 3)) # bring batch up front
        min_val = stacked_tensor.reshape(self.batch_size, -1).min(axis=-1)[:, None, None]
        max_val = stacked_tensor.reshape(self.batch_size, -1).max(axis=-1)[:, None, None]

        for i in range(len(self.transformation)):
            scores = (self.plot_memory["score"][i] - min_val) / (max_val - min_val)
            self.__plot_level(orbit_samples=self.plot_memory["orbit"][i],
                              embedding=self.plot_memory["embedding"][i],
                              candidates=self.plot_memory["n_max"][i],
                              scores=scores, line_thickness=line_thickness, fontsize=fontsize,
                              level=i, batch_index=batch_index, fig=subfigs[i])

        plt.savefig('./figures/its.pdf', bbox_inches='tight')

    def __plot_level(self, fig, orbit_samples: torch.tensor, embedding: torch.tensor, scores: torch.tensor,
                     candidates: torch.tensor, level: int, batch_index=0, line_thickness=1, fontsize=12):
        """ Plots on level of the search tree.
        :param orbit_samples: (B x K x O x Ch x H x W)
        :param embedding: (B x K x O x C)
        :param scores: (B x K x O)
        :param candidates: (B x K)
        :param level: (int) to be plotted
        :param batch_index: (int) to be plotted
        """
        ax = fig.subplots(self.top_k if level != 0 else 1, self.n_samples)
                              # figsize=(self.n_samples * 1.5, self.top_k * 2 if level != 0 else 2),
                              # gridspec_kw={'wspace': 0.0, 'hspace': 0.0})
        ax = ax if level != 0 else ax[None]

        # map scores to colors
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', '#176fc1'], N=256)
        colors = cmap(scores)

        # loop over branch stacks
        for k in range(self.top_k if level != 0 else 1):
            label_idx = np.argmax(embedding[batch_index, k], axis=-1)

            # loop over orbit samples (elements in a branch stack)
            for s in range(self.n_samples):

                # extend the image by a box above
                im = orbit_samples[batch_index, k, s].transpose((1, 2, 0))
                im = np.tile(im, (1, 1, 3)) if im.shape[-1] == 1 else im

                # some pre-processors do not map to [0,1]
                if im.min() != 0.:
                    im = (im - im.min()) / (im.max() - im.min())

                # add score-colored rectangle
                color_rec = np.tile(np.array(colors[batch_index, k, s, :3])[None, None],(5*line_thickness, im.shape[1], 1))
                im = np.concatenate([color_rec, im], axis=0)

                # plot the image
                ax[k, s].imshow(im, cmap='gray')
                ax[k, s].set_xlim(0, im.shape[1])
                ax[k, s].set_ylim(im.shape[0], 0)
                ax[k, s].axis('off')

                # add predicted label
                pred_label = str(self.labels[label_idx[s]]).split(',')[0]
                # if pred_label == '4':
                #     pred_label = '9'
                ax[k, s].text(im.shape[1] / 2, 2.5 if line_thickness == 1 else 7.,
                              pred_label, color='black', ha='center', va='center',
                              weight='bold', fontsize=fontsize)

                # highlight candidates
                if ((level == 0 and np.any(s == candidates[batch_index]))
                        or (level > 0 and s == candidates[batch_index, k])):
                        highlight_subplot(ax[k, s], plt.get_cmap('coolwarm')(1.), 1*line_thickness)
