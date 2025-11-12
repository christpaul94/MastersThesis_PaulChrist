from pykeops.torch import LazyTensor
import torch
import triton
import triton.language as tl
import numpy as np
import math

def calculate_interaction_strength(
    r0_factor: float,
    a_s_bohr: float = 98.98
) -> tuple[float, float]:
    """
    Berechnet r0_phys und C_phys für das Rb-87 Pseudopotenzial.

    Args:
        r0_factor: faktor
        a_s_bohr: s-Wellen-Streulänge in Bohr-Radien.
                   
    Returns:
        Tupel (r0_phys, C_phys)  
    """
    
    HBAR = 1.05457e-34  # J·s
    A0 = 5.29177e-11   # m (Bohr-Radius)
    MASS_RB87 = 86.909 * 1.66054e-27 # kg 
    
    # Streulänge berechnen
    a_phys = a_s_bohr * A0
    
    # Effektive Reichweite des Potenzials berechnen
    r0_phys = a_phys * r0_factor
    
    # Formel für C_phys  
    numerator_phys = 2 * HBAR**2 * a_phys
    denominator_phys = MASS_RB87 * math.sqrt(2 * math.pi) * (r0_phys**3)
    C_phys = numerator_phys / denominator_phys
    
    return r0_phys, C_phys


# ========================================================================
@torch.compile(mode="max-autotune")
def compiled_chunk_step(x_i_chunk, x_j_all, r0_2, c):
    R_vec = x_i_chunk.unsqueeze(1) - x_j_all.unsqueeze(0) # diese zeile umschreiben!!
    r_sq = torch.sum(R_vec**2, dim=2)
    exp_term = (-r_sq / (2 * r0_2)).exp()
    potential_ij = c * exp_term
    potential_sum_i_chunk = torch.sum(potential_ij, dim=1)
    force_vec_ij = potential_ij.unsqueeze(2) * R_vec / r0_2
    forces_chunk = torch.sum(force_vec_ij, dim=1)
    return forces_chunk, potential_sum_i_chunk

def softcore_forces_torch_compile(
    positions: torch.Tensor,
    r0: float,
    c: float,
    chunk_size: int = 2048 * 2
):
    N = positions.shape[0]
    r0_2 = r0**2
    forces_list = []
    potential_sum_list = []

    for x_i_chunk in positions.split(chunk_size):
        forces_chunk, potential_sum_i_chunk = compiled_chunk_step(
            x_i_chunk, positions, r0_2, c
        )
        forces_list.append(forces_chunk.clone())
        potential_sum_list.append(potential_sum_i_chunk.clone())

    forces = torch.cat(forces_list, dim=0)
    potential_sum_i = torch.cat(potential_sum_list, dim=0)
    total_sum = potential_sum_i.sum()
    total_potential = 0.5 * (total_sum - N * c)
    return forces, total_potential


# ========================================================================

def pair_keops_fp(positions: torch.Tensor, r0: float, c: float):
    N = positions.shape[0]
    r0_2 = r0**2
    x_i = LazyTensor(positions[:, None, :])
    x_j = LazyTensor(positions[None, :, :])

    diff = x_i - x_j
    r_sq = diff.sqnorm2()

    exp_term = (-r_sq / (2 * r0_2)).exp()
    potential_ij = c * exp_term

    # potential_sum_i enthält die Selbst-Interaktion (V_ii = c * exp(0) = c)
    potential_sum_i = potential_ij.sum_reduction(axis=1)
    total_sum = potential_sum_i.sum()

    # Korrektur für Selbst-Interaktion V_ii = c und Doppeltzählung
    total_potential = 0.5 * (total_sum - N * c)

    forces = (potential_ij * diff / r0_2).sum_reduction(axis=1)
    return forces, total_potential


# --- CUDA Kernel (Triton) ---
@triton.jit
def softcore_kernel_corrected(
    ptr_positions, ptr_forces, ptr_potential_sum_i,
    N: tl.int32, r0: tl.float32, c: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(axis=0)
    pos_i_x = tl.load(ptr_positions + i * 3 + 0)
    pos_i_y = tl.load(ptr_positions + i * 3 + 1)
    pos_i_z = tl.load(ptr_positions + i * 3 + 2)

    force_acc_x, force_acc_y, force_acc_z = 0.0, 0.0, 0.0
    potential_acc = 0.0
    r0_2 = r0 * r0

    for j_tile in range(0, N, BLOCK_SIZE):
        offs_j = j_tile + tl.arange(0, BLOCK_SIZE)
        mask_j = offs_j < N
        ptr_j = ptr_positions + offs_j * 3
        pos_j_x = tl.load(ptr_j + 0, mask=mask_j)
        pos_j_y = tl.load(ptr_j + 1, mask=mask_j)
        pos_j_z = tl.load(ptr_j + 2, mask=mask_j)

        diff_x = pos_i_x - pos_j_x
        diff_y = pos_i_y - pos_j_y
        diff_z = pos_i_z - pos_j_z
        r_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z

        # Eigeninteraktion (i=j) ignorieren
        r_sq = tl.where(i == offs_j, float('inf'), r_sq)

        exp_term = tl.exp(-r_sq / (2.0 * r0_2))
        exp_term = tl.where(mask_j, exp_term, 0.0)

        potential_ij = c * exp_term
        potential_acc += tl.sum(potential_ij)
        force_scalar = potential_ij / r0_2

        force_acc_x += tl.sum(force_scalar * diff_x)
        force_acc_y += tl.sum(force_scalar * diff_y)
        force_acc_z += tl.sum(force_scalar * diff_z)

    tl.store(ptr_forces + i * 3 + 0, force_acc_x)
    tl.store(ptr_forces + i * 3 + 1, force_acc_y)
    tl.store(ptr_forces + i * 3 + 2, force_acc_z)
    tl.store(ptr_potential_sum_i + i, potential_acc)

def pair_triton_fp(positions: torch.Tensor, r: float, c: float):
    N = positions.shape[0]
    forces = torch.empty_like(positions)
    potential_sum_i = torch.empty(N, device=positions.device, dtype=positions.dtype)
    grid = (N,)

    softcore_kernel_corrected[grid](
        positions, forces, potential_sum_i, N, r, c, BLOCK_SIZE=1024
    )

    total_sum = potential_sum_i.sum()
    total_potential = total_sum * 0.5

    return forces, total_potential






# ========================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_J': 128, 'num_warps': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE_J': 256, 'num_warps': 4}, num_stages=1),
        triton.Config({'BLOCK_SIZE_J': 512, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_SIZE_J': 1024, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_SIZE_J': 128, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE_J': 256, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE_J': 512, 'num_warps': 8}, num_stages=2),
        triton.Config({'BLOCK_SIZE_J': 512, 'num_warps': 16}, num_stages=2),
        triton.Config({'BLOCK_SIZE_J': 1024, 'num_warps': 16}, num_stages=2),
    ],
    key=['N', 'D'],
)
@triton.jit
def softcore_forces_triton_kernel1(
    X_ptr, Forces_ptr,
    Potential_Sum_i_ptr,
    N, D, R0_SQUARED, C_PARAM,
    D_DIM_POW2: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr
):
    i = tl.program_id(axis=0)
    offs_d = tl.arange(0, D_DIM_POW2)
    x_i_ptr = X_ptr + i * D + offs_d
    x_i = tl.load(x_i_ptr, mask=offs_d < D, other=0.0)
    x_i = tl.cast(x_i, tl.float32)

    f_i_acc = tl.zeros((D_DIM_POW2,), dtype=tl.float32)
    pot_i_acc = 0.0

    for j_start in range(0, N, BLOCK_SIZE_J):
        j_offs = j_start + tl.arange(0, BLOCK_SIZE_J)
        j_mask = j_offs < N

        x_j_ptr = X_ptr + j_offs[:, None] * D + offs_d[None, :]
        x_j = tl.load(x_j_ptr, mask=j_mask[:, None] & (offs_d[None, :] < D), other=0.0)
        x_j = tl.cast(x_j, tl.float32)

        R_vec = x_i[None, :] - x_j
        r_sq = tl.sum(R_vec * R_vec, axis=1)

        exp_term = tl.exp(-r_sq / (2.0 * R0_SQUARED))
        potential_ij_raw = C_PARAM * exp_term

        # 1. Potential-Akkumulation (KeOps-Logik: Zählt V_ii mit)
        pot_i_acc += tl.sum(tl.where(j_mask, potential_ij_raw, 0.0))

        # 2. Kraft-Akkumulation (maskiert V_ii)
        self_mask = (i != j_offs)
        force_mask = j_mask & self_mask
        potential_ij_for_force = tl.where(force_mask, potential_ij_raw, 0.0)

        force_vec_ij = potential_ij_for_force[:, None] * R_vec / R0_SQUARED
        f_i_acc += tl.sum(force_vec_ij, axis=0)

    f_i_ptr = Forces_ptr + i * D + offs_d
    tl.store(f_i_ptr, f_i_acc, mask=offs_d < D)
    pot_i_ptr = Potential_Sum_i_ptr + i
    tl.store(pot_i_ptr, pot_i_acc)

def softcore_forces_triton1(positions: torch.Tensor, r0: float, c: float):
    N, D = positions.shape
    r0_2 = r0**2
    forces = torch.empty_like(positions)
    potential_sum_i = torch.empty(N, device=positions.device, dtype=torch.float32)
    grid = (N,)
    D_DIM_POW2 = triton.next_power_of_2(D)

    softcore_forces_triton_kernel1[grid](
        positions, forces, potential_sum_i,
        N, D, r0_2, c,
        D_DIM_POW2=D_DIM_POW2
    )

    # Korrektur, da V_ii mitgezählt wurde (KeOps-Logik)
    total_sum = potential_sum_i.sum()
    total_potential = 0.5 * (total_sum - N * c)
    return forces, total_potential


# ========================================================================
