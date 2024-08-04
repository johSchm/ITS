import argparse

parser = argparse.ArgumentParser(description='Argument parser for the script.')

parser.add_argument('--gpu', type=int, default=0, help='GPU index to use (default: 0)')

parser.add_argument('--transformations', type=int, nargs='+',
                    default=['rotation', 'scaling'],
                    choices=['rotation', 'scaling', 'shearing', 'translation'],
                    help='List of transformations to apply.')

parser.add_argument('--domains', type=float, nargs='+', action='append',
                    default=[(3.1415,), (0.5,)],
                    help='List of domains for each transformation.')

parser.add_argument('--n_samples', type=int, default=17, help='Number of samples (default: 17)')

parser.add_argument('--model_path', type=str, default='./model/mnist.pth',
                    help='Path to the model file (default: ./model/mnist.pth)')

parser.add_argument('--test_set_loader_path', type=str, default='./data/mnist/test_loader.pickle',
                    help='Path to the test set loader file (default: ./data/mnist/test_loader.pickle)')

parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist'],
                    help='Dataset to use (default: mnist)')

parser.add_argument('--n_hypotheses', type=int, default=3, help='Number of hypotheses (default: 3)')

parser.add_argument('--mc_steps', type=int, default=10, help='Number of Monte Carlo steps (default: 10)')

parser.add_argument('--batch_size', type=int, default=128, help='The batch size used for testing. (default: 128)')

parser.add_argument('--change_of_mind', type=str, default='score', choices=['score', 'off'],
                    help="Criterion for change of mind (default: 'score')")

parser.add_argument('--en_unique_class_condition', type=bool, default=True,
                    help='Enable unique class condition (default: True)')

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torchvision
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from src.transform import AffineTransformation, multi_transform
from src.search import InverseTransformationSearch

parsed_transformations = []
for T in args.transformations:
    if T == 'rotation':
        parsed_transformations.append(AffineTransformation.ROTATION.value)
    elif T == 'scaling':
        parsed_transformations.append(AffineTransformation.SCALING.value)
    elif T == 'shearing':
        parsed_transformations.append(AffineTransformation.SHEARING.value)
    elif T == 'translation':
        parsed_transformations.append(AffineTransformation.TRANSLATION.value)
    else:
        raise NotImplementedError("Transformation '{}' not implemented (yet).".format(T))
args.transformations = parsed_transformations


def random_transform(x):
    # todo hotfix pseudo-batch
    n = torch.randint(0, args.n_samples, (2, len(args.transformations)), device=x.device)
    x, _, _ = multi_transform(x[None].repeat(2, 1, 1, 1), args.transformations, n,
                              n_samples=args.n_samples, domain=args.domains)
    return x[0]


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    random_transform
])

if args.dataset == 'mnist':
    # essentially excluding either 6 or 9 as these would cause degenerations under 180 degree rotation
    from src.data import load_mnist
    n_classes = 9
    test_set = load_mnist(test_set=True, transform=transform)
elif args.dataset == 'fmnist':
    n_classes = 10
    test_set = torchvision.datasets.FashionMNIST('./data/', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

model = torch.jit.load(args.model_path)

its = InverseTransformationSearch(model,
                                  transformation=args.transformations,
                                  domain=args.domains,
                                  n_samples=args.n_samples,
                                  n_hypotheses=args.n_hypotheses,
                                  mc_steps=args.mc_steps,
                                  change_of_mind=args.change_of_mind,
                                  en_unique_class_condition=args.en_unique_class_condition)

model.eval()
test_acc = 0
for d, (data, target) in enumerate(test_loader):
    data, target = data.cuda(), target.cuda()
    data_canonic, embedding = its.infer(data)
    output = embedding.argmax(dim=-1)[:, 0]
    test_acc += output.eq(target).sum().item()
test_acc /= len(test_set)
print(f'Mean accuracy on the transformed test set [with ITS]: {test_acc}.')
