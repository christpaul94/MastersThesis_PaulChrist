import torch
import triton
import triton.language as tl
import time
import numpy as np
import torch
import triton
import triton.language as tl
import time
from typing import Dict, Callable
import numpy as np
import torch
import triton
import triton.language as tl
import time
from typing import Dict
import numpy as np

import torch
import triton
import triton.language as tl
import time
from typing import Dict


dtype = torch.float32 # HINWEIS: Wird jetzt von den meisten Funktionen nicht mehr global verwendet
torch.manual_seed(42)

# ==============================================================================
# PHASEN 1-3: CELL-LIST-ALGORITHMUS ZUM SORTIEREN
# ==============================================================================

@triton.jit(noinline=True)
def assign_cells_kernel(
    ptr_positions, ptr_particle_cell_indices,
    box_min_x, box_min_y, box_min_z, cell_size,
    grid_dim_x, grid_dim_y, grid_dim_z,
    N: tl.int32, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    pos_x = tl.load(ptr_positions + offset * 3 + 0, mask=mask)
    pos_y = tl.load(ptr_positions + offset * 3 + 1, mask=mask)
    pos_z = tl.load(ptr_positions + offset * 3 + 2, mask=mask)
    cx = tl.floor((pos_x - box_min_x) / cell_size).to(tl.int32)
    cy = tl.floor((pos_y - box_min_y) / cell_size).to(tl.int32)
    cz = tl.floor((pos_z - box_min_z) / cell_size).to(tl.int32)
    cx = tl.maximum(0, tl.minimum(cx, grid_dim_x - 1))
    cy = tl.maximum(0, tl.minimum(cy, grid_dim_y - 1))
    cz = tl.maximum(0, tl.minimum(cz, grid_dim_z - 1))
    cell_idx = (cx * grid_dim_y + cy) * grid_dim_z + cz
    tl.store(ptr_particle_cell_indices + offset, cell_idx, mask=mask)

def phase1_assign_particles_to_cells(positions: torch.Tensor, cell_size: float):
    N = positions.shape[0]
    box_min = positions.min(dim=0).values
    box_max = positions.max(dim=0).values
    box_size = (box_max - box_min) + 1e-6
    grid_dims = (torch.ceil(box_size / cell_size)).to(torch.int32)
    num_cells = torch.prod(grid_dims).item()
    
    # GEÄNDERT: device von `positions` abgeleitet
    particle_cell_indices = torch.empty(N, dtype=torch.int32, device=positions.device) 
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    assign_cells_kernel[grid](
        positions, particle_cell_indices,
        box_min[0].item(), box_min[1].item(), box_min[2].item(),
        cell_size,
        grid_dims[0].item(), grid_dims[1].item(), grid_dims[2].item(),
        N, BLOCK_SIZE=BLOCK_SIZE
    )
    return particle_cell_indices, grid_dims, num_cells

def phase2_sort_particles_by_cell(positions: torch.Tensor, particle_cell_indices: torch.Tensor):
    _, sorted_indices = torch.sort(particle_cell_indices)
    sorted_positions = positions[sorted_indices]
    return sorted_indices, sorted_positions

def phase3_find_cell_boundaries(particle_cell_indices: torch.Tensor, num_cells: int):
    # GEÄNDERT: device von `particle_cell_indices` abgeleitet
    cell_offsets = torch.zeros(num_cells + 1, dtype=torch.int32, device=particle_cell_indices.device)
    
    true_counts = torch.bincount(particle_cell_indices, minlength=num_cells)
    cell_offsets[1:] = torch.cumsum(true_counts, dim=0, dtype=torch.int32)
    return cell_offsets

# ==============================================================================
# PHASE 4: VERLET-LISTEN-AUFBAU
# ==============================================================================

@triton.jit(noinline=True)
def count_neighbors_kernel(
    ptr_sorted_positions, ptr_cell_offsets,
    ptr_neighbor_counts,
    grid_dim_x, grid_dim_y, grid_dim_z,
    r_verlet_sq, N: tl.int32,
):
    c_i = tl.program_id(axis=0)
    start_i = tl.load(ptr_cell_offsets + c_i)
    end_i = tl.load(ptr_cell_offsets + c_i + 1)
    num_particles_in_cell = end_i - start_i

    cz = c_i % grid_dim_z
    cy = (c_i // grid_dim_z) % grid_dim_y
    cx = c_i // (grid_dim_y * grid_dim_z)

    for i_offset in range(num_particles_in_cell):
        idx_i_sorted = start_i + i_offset

        pos_i_x = tl.load(ptr_sorted_positions + idx_i_sorted * 3 + 0)
        pos_i_y = tl.load(ptr_sorted_positions + idx_i_sorted * 3 + 1)
        pos_i_z = tl.load(ptr_sorted_positions + idx_i_sorted * 3 + 2)

        count = 0
        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                for dcz in range(-1, 2):
                    ncx = cx + dcx
                    ncy = cy + dcy
                    ncz = cz + dcz

                    if ncx >= 0 and ncx < grid_dim_x and ncy >= 0 and ncy < grid_dim_y and ncz >= 0 and ncz < grid_dim_z:
                        c_j = (ncx * grid_dim_y + ncy) * grid_dim_z + ncz
                        start_j = tl.load(ptr_cell_offsets + c_j)
                        end_j = tl.load(ptr_cell_offsets + c_j + 1)

                        for idx_j_sorted in range(start_j, end_j):
                            if idx_i_sorted != idx_j_sorted:
                                pos_j_x = tl.load(ptr_sorted_positions + idx_j_sorted * 3 + 0)
                                pos_j_y = tl.load(ptr_sorted_positions + idx_j_sorted * 3 + 1)
                                pos_j_z = tl.load(ptr_sorted_positions + idx_j_sorted * 3 + 2)
                                dist_sq = (pos_i_x - pos_j_x)*(pos_i_x - pos_j_x) + (pos_i_y - pos_j_y)*(pos_i_y - pos_j_y) + (pos_i_z - pos_j_z)*(pos_i_z - pos_j_z)
                                if dist_sq < r_verlet_sq:
                                    count += 1

        tl.store(ptr_neighbor_counts + idx_i_sorted, count)

@triton.jit(noinline=True)
def fill_verlet_list_kernel(
    ptr_sorted_positions, ptr_cell_offsets, ptr_verlet_offsets,
    ptr_verlet_indices,
    grid_dim_x, grid_dim_y, grid_dim_z,
    r_verlet_sq, N: tl.int32,
):
    c_i = tl.program_id(axis=0)
    start_i = tl.load(ptr_cell_offsets + c_i)
    end_i = tl.load(ptr_cell_offsets + c_i + 1)
    num_particles_in_cell = end_i - start_i

    cz = c_i % grid_dim_z
    cy = (c_i // grid_dim_z) % grid_dim_y
    cx = c_i // (grid_dim_y * grid_dim_z)

    for i_offset in range(num_particles_in_cell):
        idx_i_sorted = start_i + i_offset
        pos_i_x = tl.load(ptr_sorted_positions + idx_i_sorted * 3 + 0)
        pos_i_y = tl.load(ptr_sorted_positions + idx_i_sorted * 3 + 1)
        pos_i_z = tl.load(ptr_sorted_positions + idx_i_sorted * 3 + 2)
        current_offset = tl.load(ptr_verlet_offsets + idx_i_sorted)

        for dcx in range(-1, 2):
            for dcy in range(-1, 2):
                for dcz in range(-1, 2):
                    ncx = cx + dcx
                    ncy = cy + dcy
                    ncz = cz + dcz
                    if ncx >= 0 and ncx < grid_dim_x and ncy >= 0 and ncy < grid_dim_y and ncz >= 0 and ncz < grid_dim_z:
                        c_j = (ncx * grid_dim_y + ncy) * grid_dim_z + ncz
                        start_j = tl.load(ptr_cell_offsets + c_j)
                        end_j = tl.load(ptr_cell_offsets + c_j + 1)
                        for idx_j_sorted_neighbor in range(start_j, end_j):
                            if idx_i_sorted != idx_j_sorted_neighbor:
                                pos_j_x = tl.load(ptr_sorted_positions + idx_j_sorted_neighbor * 3 + 0)
                                pos_j_y = tl.load(ptr_sorted_positions + idx_j_sorted_neighbor * 3 + 1)
                                pos_j_z = tl.load(ptr_sorted_positions + idx_j_sorted_neighbor * 3 + 2)
                                dist_sq = (pos_i_x - pos_j_x)*(pos_i_x - pos_j_x) + (pos_i_y - pos_j_y)*(pos_i_y - pos_j_y) + (pos_i_z - pos_j_z)*(pos_i_z - pos_j_z)
                                if dist_sq < r_verlet_sq:
                                    tl.store(ptr_verlet_indices + current_offset, idx_j_sorted_neighbor)
                                    current_offset += 1

# GEÄNDERT: Signatur angepasst, um n_particles zu akzeptieren
def build_verlet_list(positions, r_verlet, r_verlet_sq, n_particles):
    cell_size = r_verlet
    
    # device wird von `positions` abgeleitet
    current_device = positions.device 
    
    pci, grid_dims, num_cells = phase1_assign_particles_to_cells(positions, cell_size)
    sorted_indices, sorted_positions = phase2_sort_particles_by_cell(positions, pci)
    cell_offsets = phase3_find_cell_boundaries(pci, num_cells)
    
    # GEÄNDERT: n_particles und device verwendet
    neighbor_counts_sorted = torch.empty(n_particles, dtype=torch.int32, device=current_device)
    
    grid = (num_cells,)
    count_neighbors_kernel[grid](
        sorted_positions, cell_offsets,
        neighbor_counts_sorted,
        grid_dims[0].item(), grid_dims[1].item(), grid_dims[2].item(),
        r_verlet_sq, 
        n_particles # GEÄNDERT: n_particles übergeben
    )
    
    # GEÄNDERT: n_particles und device verwendet
    verlet_offsets_sorted = torch.zeros(n_particles + 1, dtype=torch.int32, device=current_device)
    
    verlet_offsets_sorted[1:] = torch.cumsum(neighbor_counts_sorted, dim=0)
    total_neighbors = verlet_offsets_sorted[-1].item()
    
    # GEÄNDERT: device verwendet
    verlet_indices_sorted = torch.empty(total_neighbors, dtype=torch.int32, device=current_device)
    
    fill_verlet_list_kernel[grid](
        sorted_positions, cell_offsets, verlet_offsets_sorted,
        verlet_indices_sorted,
        grid_dims[0].item(), grid_dims[1].item(), grid_dims[2].item(),
        r_verlet_sq, 
        n_particles # GEÄNDERT: n_particles übergeben
    )
    return verlet_indices_sorted, verlet_offsets_sorted, sorted_indices

# ==============================================================================
# PHASE 5: KRAFT- UND POTENZIAL-KERNEL MIT VERLET-LISTEN
# ==============================================================================

@triton.jit
def force_and_potential_kernel_with_verlet_list(
    ptr_positions, ptr_verlet_indices, ptr_verlet_offsets,
    ptr_forces, ptr_potential_per_i,
    r_cutoff_sq, r0, c, N: tl.int32
):
    pid = tl.program_id(axis=0)
    if pid >= N:
        return

    pos_i_x = tl.load(ptr_positions + pid * 3 + 0)
    pos_i_y = tl.load(ptr_positions + pid * 3 + 1)
    pos_i_z = tl.load(ptr_positions + pid * 3 + 2)

    force_acc_x = 0.0
    force_acc_y = 0.0
    force_acc_z = 0.0
    potential_acc = 0.0
    r0_2 = r0 * r0

    start = tl.load(ptr_verlet_offsets + pid)
    end = tl.load(ptr_verlet_offsets + pid + 1)

    for k in range(start, end):
        idx_j = tl.load(ptr_verlet_indices + k)
        pos_j_x = tl.load(ptr_positions + idx_j * 3 + 0)
        pos_j_y = tl.load(ptr_positions + idx_j * 3 + 1)
        pos_j_z = tl.load(ptr_positions + idx_j * 3 + 2)
        diff_x = pos_i_x - pos_j_x
        diff_y = pos_i_y - pos_j_y
        diff_z = pos_i_z - pos_j_z
        r_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z

        if r_sq < r_cutoff_sq and r_sq > 1e-9:
            exp_term = tl.exp(-r_sq / (2.0 * r0_2))
            potential_acc += c * exp_term
            force_scalar = c * exp_term / r0_2
            force_acc_x += force_scalar * diff_x
            force_acc_y += force_scalar * diff_y
            force_acc_z += force_scalar * diff_z

    tl.store(ptr_forces + pid * 3 + 0, force_acc_x)
    tl.store(ptr_forces + pid * 3 + 1, force_acc_y)
    tl.store(ptr_forces + pid * 3 + 2, force_acc_z)
    tl.store(ptr_potential_per_i + pid, potential_acc)

# GEÄNDERT: Signatur angepasst, um n_particles zu akzeptieren
def calculate_forces_and_potential_verlet(positions, verlet_indices, verlet_offsets, sorted_indices, r_cutoff_sq, r0, c, n_particles):
    sorted_positions = positions[sorted_indices]
    forces_sorted = torch.zeros_like(sorted_positions)
    
    # GEÄNDERT: n_particles und device/dtype von `positions` abgeleitet
    potential_per_i_sorted = torch.zeros(n_particles, dtype=positions.dtype, device=positions.device)
    
    # GEÄNDERT: n_particles verwendet
    grid = (n_particles,) 
    
    force_and_potential_kernel_with_verlet_list[grid](
        sorted_positions, verlet_indices, verlet_offsets,
        forces_sorted, potential_per_i_sorted,
        r_cutoff_sq, r0, c, 
        n_particles # GEÄNDERT: n_particles übergeben
    )
    forces = torch.zeros_like(positions)
    forces[sorted_indices] = forces_sorted
    total_potential = 0.5 * potential_per_i_sorted.sum()
    return forces, total_potential

# ==============================================================================
# ANGEPASSTER VERLET-INTEGRATOR MIT VERLET-LISTEN
# ==============================================================================
def run_verlet_list_simulation_general(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    trap_force_func: Callable,
    trap_force_params: Dict,
    pair_force_params: Dict,
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    substeps: int = 100,
    silent: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Führt eine Simulation mit dem Velocity Verlet-Algorithmus durch,
    der eine Verlet-Liste für die Paar-Wechselwirkungen verwendet.
    Akzeptiert eine generische Fallenkraft-Funktion.
    Verwendet keine globalen Variablen.
    """
    with torch.no_grad():
        # n_particles ist jetzt eine lokale Variable
        num_save_points, n_particles = t_values.size(0), q0.size(0)
        
        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_trap_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Verlet-Parameter-Extraktion ---
        try:
            r_cutoff = pair_force_params['r_cutoff']
            r_skin = pair_force_params.get('r_skin', r_cutoff * 0.2)
            r0 = pair_force_params['r0']
            c = pair_force_params['c']
        except KeyError as e:
            print(f"Fehler: Notwendiger Parameter {e} nicht in `pair_force_params` gefunden.")
            print("`pair_force_params` muss 'r_cutoff', 'r0' und 'c' enthalten.")
            return {}
        
        r_verlet = r_cutoff + r_skin
        r_verlet_sq = r_verlet ** 2
        r_cutoff_sq = r_cutoff ** 2
        
        if not silent:
            print(f"Verlet-Parameter: r0={r0:.2f}, c={c:.2f}, r_cutoff={r_cutoff:.2f}, r_skin={r_skin:.2f}")

        # Initialer Verlet-List Build
        # Übergibt n_particles
        verlet_indices, verlet_offsets, sorted_indices = build_verlet_list(q_current, r_verlet, r_verlet_sq, n_particles)
        positions_at_last_build = q_current.clone()
        
        # Initialisierung
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        
        f_trap_current, pot_trap, _ = trap_force_func(q_current, **trap_force_params)
        # Übergibt n_particles
        f_p_current, pot_p = calculate_forces_and_potential_verlet(q_current, verlet_indices, verlet_offsets, sorted_indices, r_cutoff_sq, r0, c, n_particles)
        
        a_current = f_trap_current + f_p_current
        potential_trap_out[0], potential_pair_out[0] = pot_trap, pot_p

        p_half = torch.empty_like(p_current)
        q_next = torch.empty_like(q_current)
        a_next = torch.empty_like(a_current)
        p_next = torch.empty_like(p_current)

        # Hauptschleife
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps

            pot_trap_next = pot_trap
            pot_p_next = pot_p

            # Innere Schleife
            for j in range(substeps):
                p_half = p_current + 0.5 * dt * a_current
                q_next = q_current + dt * p_half

                # Rebuild-Check
                displacements = torch.sqrt(torch.sum((q_next - positions_at_last_build)**2, dim=1))
                if torch.max(displacements) > r_skin / 2.0:
                    # Übergibt n_particles
                    verlet_indices, verlet_offsets, sorted_indices = build_verlet_list(q_next, r_verlet, r_verlet_sq, n_particles)
                    positions_at_last_build = q_next.clone()
                
                f_trap_next, pot_trap_next, _ = trap_force_func(q_next, **trap_force_params)
                # Übergibt n_particles
                f_p_next, pot_p_next = calculate_forces_and_potential_verlet(q_next, verlet_indices, verlet_offsets, sorted_indices, r_cutoff_sq, r0, c, n_particles)
                a_next = f_trap_next + f_p_next

                p_next = p_half + 0.5 * dt * a_next

                q_current, p_current, a_current = q_next, p_next, a_next

            # Speichern
            q_out[i], p_out[i] = q_current, p_current
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_trap_out[i] = pot_trap_next
            potential_pair_out[i] = pot_p_next

            if not silent:
                loop_end_time = time.perf_counter()
                eta_seconds = int((loop_end_time - loop_start_time) * (num_save_points - 1 - i))
                print(f"\rIntegration {100 * (i + 1) / num_save_points:.0f}% "
                      f"| ETA: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)

        if not silent:
            print("\nIntegration abgeschlossen.")
            
        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_trap": potential_trap_out,
            "potential_energy_pair": potential_pair_out
        }
