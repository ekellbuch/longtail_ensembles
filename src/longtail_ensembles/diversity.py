import numpy as np

def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = np.argmax(logits_1, axis=-1)
  preds_2 = np.argmax(logits_2, axis=-1)
  return preds_1 != preds_2

def cosine_distance(x, y):
  """Cosine distance between vectors x and y."""
  x_norm = np.sqrt(np.sum(np.power(x, 2), axis=-1))
  x_norm = np.reshape(x_norm, (-1, 1))
  y_norm = np.sqrt(np.sum(np.power(y, 2), axis=-1))
  y_norm = np.reshape(y_norm, (-1, 1))
  normalized_x = x / x_norm
  normalized_y = y / y_norm
  return np.sum(normalized_x * normalized_y, axis=-1)

def kl_divergence(p, q):
  return np.sum(p * np.log(p / q), axis=-1)


diversity_metrics = {'avg_disagreement': disagreement,
                    'cosine_similarity': cosine_distance,
                    'kl_divergence': kl_divergence}


if __name__ == "__main__":
  pass