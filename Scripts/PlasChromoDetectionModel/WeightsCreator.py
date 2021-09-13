import torch
from datetime import datetime


def make_weights_for_balanced_classes(nclasses, labels):
    count = torch.bincount(labels, minlength=nclasses)
    print(f'Calculating weights for {nclasses} classes....')
    N = torch.sum(count)
    weight_per_class = torch.mul(torch.reciprocal(count.float()), N)
    weight_per_class[weight_per_class == float("Inf")] = 0
    weights = torch.zeros(len(labels), dtype=torch.float)
    print('Assigning class weights....')

    for idx, val in enumerate(labels):
        weights[idx] = weight_per_class[val]

    print('==== Weights calculated ====')

    return weights
