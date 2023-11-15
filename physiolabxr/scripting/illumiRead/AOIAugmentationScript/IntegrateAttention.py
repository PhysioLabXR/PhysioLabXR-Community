from typing import Union

import numpy as np
import torch

def integrate_attention(attention_human: Union[np.ndarray, torch.Tensor], attention_vit: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    integrates the vit attention of between different patches based on the how much attention the human is paying
    to each patch.
    Essentially we take each patch as a query. Look at how the n keys attends to it. This is a vector of length n.
    Then we weigh this vector by the human attention at this patch. We do so for all patches as queries so we end up with
    n size n vectors. We sum them up to get a single vector of size n. This is the integrated attention.
    @param attention_human: array of shape (num_patches)
    @param attention_vit: array of shape (num_patches+1, num_patches+1) or (num_patches, num_patches). Plus 1 is for the
        class token. The class token is not used in the integration.
    @return:
    """
    assert len(attention_human.shape) == 1, f'attention_human should be 1D, but is {attention_human.shape}'
    assert len(attention_vit.shape) == 2, f'attention_vit should be 2D, but is {attention_vit.shape}'
    assert attention_human.shape[0] == attention_vit.shape[0] - 1 or attention_human.shape[0] == attention_vit.shape[0], \
        f"attention shape mismatch: {attention_human.shape[0]} vs {attention_vit.shape[0]}. attention_vit's shape should be (num_patches+1, num_patches+1) or (num_patches, num_patches). Plus 1 is for the class token. The class token is not used in the integration."
    # type check
    assert type(attention_human) == np.ndarray or type(attention_human) == torch.Tensor, \
        f"attention_human should be either np.ndarray or torch.Tensor, but is {type(attention_human)}"
    assert type(attention_vit) == np.ndarray or type(attention_vit) == torch.Tensor, \
        f"attention_vit should be either np.ndarray or torch.Tensor, but is {type(attention_human)}"
    # all convert to torch
    if type(attention_human) == np.ndarray:
        attention_human = torch.from_numpy(attention_human)
    if type(attention_vit) == np.ndarray:
        attention_vit = torch.from_numpy(attention_vit)

    if attention_human.shape[0] == attention_vit.shape[0] - 1:
        attention_vit = attention_vit[:-1, :-1]  # discard the class token

    # convert to float32 if not already
    if attention_human.dtype != torch.float32:
        attention_human = attention_human.float()
    if attention_vit.dtype != torch.float32:
        attention_vit = attention_vit.float()

    # use np einsum or torch einsum based on input type
    return torch.einsum('i,ij->j', attention_human, attention_vit).detach().cpu().numpy()

def test_integrate_attention():
    attention_human = np.array([0.1, 0.2, 0.3, 0.4])
    attention_vit = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]], dtype=float)
    expected_results = 0.1 * np.array([1, 2, 3, 4], dtype=float) + 0.2 * np.array([5, 6, 7, 8], dtype=float) + 0.3 * np.array(
        [9, 10, 11, 12], dtype=float) + 0.4 * np.array([13, 14, 15, 16], dtype=float)
    results = integrate_attention(attention_human, attention_vit)
    assert np.allclose(results, expected_results), f"results: {results}, expected_results: {expected_results}"

if __name__ == '__main__':
    test_integrate_attention()
    print('test passed')