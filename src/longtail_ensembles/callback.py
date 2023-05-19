from pytorch_lightning.callbacks import Callback

class GradNormCallbackSplit(Callback):
    """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

    def on_after_backward(self, trainer, model):
        model.log("model/grad_norm", gradient_norm(model))
        has_models = getattr(model, "models", 0)
        if has_models != 0:
            for model_idx, model_ in enumerate(model.models):
                model.log(f"model/grad_norm_{model_idx}",
                          gradient_norm(model_))


class GradNormCallback(Callback):
    """
  Logs the gradient norm.
  Edited from https://github.com/Lightning-AI/lightning/issues/1462
  """

    def on_after_backward(self, trainer, model):
        model.log("my_model/grad_norm", gradient_norm(model))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**(1. / 2)
    return total_norm
