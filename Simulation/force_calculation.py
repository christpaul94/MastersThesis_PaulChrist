from pykeops.torch import LazyTensor

def pair_keops_fp(positions: torch.Tensor, r: float, c: float):
    N = positions.shape[0]
    r0_2 = r**2
    x_i = LazyTensor(positions[:, None, :]); x_j = LazyTensor(positions[None, :, :])
    diff = x_i - x_j
    r_sq = diff.sqnorm2()
    exp_term = (-r_sq / (2 * r0_2)).exp()
    potential_ij = c * exp_term
    total_sum = potential_ij.sum_reduction(axis=1).sum()
    total_potential = 0.5 * (total_sum - N * c)
    forces = (potential_ij * diff / r0_2).sum_reduction(axis=1)
    return forces, total_potential
