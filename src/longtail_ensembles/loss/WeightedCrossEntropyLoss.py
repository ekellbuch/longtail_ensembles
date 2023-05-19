import torch.nn as nn


class WeightedCrossEntropy(nn.Module):
    """
    Weighted cross entropy loss L = - w_y log (p_y)  for class y
    where  w_y = n / (k * n_y)

    n is the number of samples,
    n_y is the sample number of class y.
    k is the number of classes
    To down-weight majority class
    - loss_weight * log (exp()/sum(exp()))
    """
    def __init__(self, sample_per_class, **kwargs):
        super(WeightedCrossEntropy, self).__init__()
        # calculate weight
        weights = sample_per_class.sum() / (len(sample_per_class) * sample_per_class)
        self.loss_weight = weights
        self.kwargs = kwargs

    def forward(self, output, target):
        weight = self.loss_weight.type_as(output)
        loss = nn.functional.cross_entropy(output, target,
                                           weight=weight,
                                           reduction='mean',
                                           **self.kwargs)
        return loss
