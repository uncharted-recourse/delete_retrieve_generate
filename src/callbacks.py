
import numpy as np 

def mean_masked_entropy(preds, mask = None):

    if mask is None:
        return np.mean(-np.sum(probs * np.log(probs + 1e-8), axis=-1))

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    entropy *= mask  # NOTE mask is 0 at positions to ignore
    # average over unmasked sequence positions, then over samples
    return np.mean(np.sum(entropy, axis=-1) / np.sum(mask, axis=-1))
