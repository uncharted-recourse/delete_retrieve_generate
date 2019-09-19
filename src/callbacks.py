
import numpy as np 

def mean_masked_entropy(probs, y_true, pad_id):
    """ calculate the mean entropy of a (masked) probability distribution -> probs * log(probs)"""

    mask = np.not_equal(y_true, pad_id)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    entropy *= mask  # NOTE mask is 0 at positions to ignore
    # average over unmasked sequence positions, then over samples
    return np.mean(np.sum(entropy, axis=-1) / np.sum(mask, axis=-1))
