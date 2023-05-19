import torch.nn as nn


class WeightedSoftmax(nn.Module):
    """
    Weighted softmax loss L = - w_y log (p_y)  for class y
    where w_y =  1 / pi_y

    pi_y = n_y / n
    n is the number of samples,
    n_y is the sample number of class y.

    - loss_weight * log (exp()/sum(exp()))
    """
    def __init__(self, sample_per_class, **kwargs):
        super(WeightedSoftmax, self).__init__()
        # calculate weight
        self.loss_weight = sample_per_class.sum() / sample_per_class
        self.kwargs = kwargs

    def forward(self, output, target):
        weight = self.loss_weight.type_as(output)
        loss = nn.functional.cross_entropy(output, target,
                                           weight=weight,
                                           reduction='mean',
                                           **self.kwargs)
        return loss
