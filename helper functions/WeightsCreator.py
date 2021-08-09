import torch

def make_weights_for_balanced_classes(nclasses, labels):
    count = torch.bincount(labels, minlength=nclasses)
    N = torch.sum(count)
    weight_per_class = torch.mul(torch.reciprocal(count), N)
    weight_per_class[weight_per_class == float("Inf")] = 0
    weights = torch.zeros(len(labels), dtype=torch.float)
    for idx, val in enumerate(labels):
        weights[idx] = weight_per_class[val]
    return weights
