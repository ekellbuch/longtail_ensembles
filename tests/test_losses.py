from absl.testing import absltest
from absl.testing import parameterized
from torch.nn import CrossEntropyLoss

from longtail_ensembles.loss.WeightedSoftmaxLoss import WeightedSoftmax
from longtail_ensembles.loss.WeightedCrossEntropyLoss import WeightedCrossEntropy
from longtail_ensembles.loss.BalancedSoftmaxLoss import BalancedSoftmax
from longtail_ensembles.loss.FocalLoss import FocalLoss
import torch


class ModuleTestLoss(parameterized.TestCase):

  @parameterized.named_parameters(
    ("softmax_imbal", "softmax",),
    ("weighted_imbal", "weighted_softmax",),
    ("weighted_ce_imbal", "weighted_ce",),
    ("balanced_softmax_imbal", "balanced_softmax",),
    ("focal_loss_imbal", "focal_loss",),
  )
  def test_imbalanced_losses(self, loss_name):

    weights = torch.tensor([3, 3, 4], dtype=torch.float32)
    total_samples = int(weights.sum().item())
    num_classes = len(weights)
    loss_params = {}
    if loss_name == 'softmax':
      loss_function = CrossEntropyLoss
    elif loss_name == 'weighted_softmax':
      loss_function = WeightedSoftmax
      loss_params['sample_per_class'] = weights
    elif loss_name == 'weighted_ce':
      loss_function = WeightedCrossEntropy
      loss_params['sample_per_class'] = weights
    elif loss_name == 'balanced_softmax':
      loss_function = BalancedSoftmax
      loss_params['sample_per_class'] = weights
    elif loss_name == 'focal_loss':
      loss_function = FocalLoss
      loss_params['gamma'] = 0
      loss_params['size_average'] = True
    else:
      raise NotImplementedError('Unknown loss function: {}'.format(loss_name))

    # Create an instance of your loss function
    loss_fn = loss_function(**loss_params)
    # Generate some dummy data for testing:
    output = torch.randn(total_samples, num_classes)
    target = torch.randint(0, num_classes, (total_samples,))

    # Compute the loss
    loss = loss_fn(output, target)
    # manually compute the loss
    # L = - 1 / pi_y
    softmax = torch.nn.Softmax(dim=1)
    if loss_name == 'softmax':
      softmax = torch.nn.Softmax(dim=1)
      true_loss = - torch.log(softmax(output))
      true_loss = true_loss[torch.arange(total_samples), target].mean()
    elif loss_name == 'weighted_softmax':
      # L = - 1/pi_y log (p_y)
      # where pi_y = n_y / n, and n_y is the sample number of class y.
      loss_weight = weights.sum() / weights  # 1/pi_y = n / n_y
      true_loss = - loss_weight * torch.log(softmax(output))
      true_loss = true_loss[torch.arange(total_samples), target]
      true_loss = true_loss.sum() / loss_weight[target].sum()
    elif loss_name == 'weighted_ce':
      # w_y = n / (k * n_y)
      # L = sum( - w_y log (p_y)) / sum(w_y)
      loss_weight = weights.sum() / (len(weights) * weights)
      true_loss = - loss_weight * torch.log(softmax(output))
      true_loss = true_loss[torch.arange(total_samples), target]
      true_loss = true_loss.sum()/loss_weight[target].sum()
    elif loss_name == 'balanced_softmax':
      # L = - log ( (pi_y exp(z_y)) / sum(pi_j exp(z_j))
      loss_weight = weights/ weights.sum()
      true_loss = - torch.log((loss_weight * torch.exp(output))/ ((loss_weight*torch.exp(output)).sum(-1).unsqueeze(-1)))
      true_loss = true_loss[torch.arange(total_samples), target]
      true_loss = true_loss.mean()
    elif loss_name == 'focal_loss':
      # L = - (1 - p_y) ^ gamma log(p_y)
      # where p_y = exp(z_y) / sum(exp(z_j))
      true_loss = - (1 - softmax(output)) ** 0 * torch.log(softmax(output))
      true_loss = true_loss[torch.arange(total_samples), target].mean()

    self.assertAlmostEqual(loss.item(), true_loss.item(), places=5)

  @parameterized.named_parameters(
    ("softmax_bal", "softmax",),
    ("weighted_softmax_bal", "weighted_softmax",),
    ("weighted_ce_bal", "weighted_ce",),
    ("balanced_softmax_bal", "balanced_softmax",),
    ("focal_loss_bal", "focal_loss",),

  )
  def test_ifbalanced(self, loss_name):
    # Tests that all losses are the same as the cross entropy loss
    # when the weights are 1
    weights = torch.tensor([1, 1, 1], dtype=torch.float32)
    total_samples = int(weights.sum().item())
    num_classes = len(weights)
    loss_params = {}
    if loss_name == 'softmax':
      loss_function = CrossEntropyLoss
    elif loss_name == 'weighted_softmax':
      loss_function = WeightedSoftmax
      loss_params['sample_per_class'] = weights
    elif loss_name == 'weighted_ce':
      loss_function = WeightedCrossEntropy
      loss_params['sample_per_class'] = weights
    elif loss_name == 'balanced_softmax':
      loss_function = BalancedSoftmax
      loss_params['sample_per_class'] = weights
    elif loss_name == 'focal_loss':
      loss_function = FocalLoss
      loss_params['gamma'] = 0
      loss_params['size_average'] = True
    else:
      raise NotImplementedError('Unknown loss function: {}'.format(loss_name))

    # Create an instance of your loss function
    loss_fn = loss_function(**loss_params)

    # Generate some dummy data for testing:
    output = torch.randn(total_samples, num_classes, requires_grad=True)
    target = torch.randint(0, num_classes, (total_samples,))

    # Compute balanced loss
    b_loss = CrossEntropyLoss(weight=weights)
    bal_loss = b_loss(output, target)

    # Compute the loss
    loss = loss_fn(output, target)

    # compute grad for loss
    bal_loss.backward()
    grad_bal_loss = output.grad.clone()
    output.grad.zero_()  # reset gradient

    loss.backward()
    grad_loss = output.grad.clone()

    self.assertAlmostEqual(bal_loss.item(), loss.item(), places=5)
    self.assertTrue(torch.allclose(grad_bal_loss, grad_loss))


if __name__ == '__main__':
  absltest.main()