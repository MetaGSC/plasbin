def make_weights_for_balanced_classes(nclasses, labels):
        count = [0] * nclasses
        nlabels = len(labels)
        for k in range(nlabels):
            lbl_ind = int(labels[k])
            count[lbl_ind] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * nlabels
        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val]
        return weight