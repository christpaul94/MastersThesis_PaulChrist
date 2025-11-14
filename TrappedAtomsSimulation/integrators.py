import torch
from typing import Dict, Tuple, Callable
import time
import torch
from typing import Dict, Tuple, Callable
import time



### Harmonic oscillator Teil 

def harmonic_fp(
    q: torch.Tensor,
    omega_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega_squared = omega_matrix @ omega_matrix.T
    forces = - q @ omega_squared
    potential_per_particle = 0.5 * torch.einsum('ni,ij,nj->n', q, omega_squared, q)
    total_potential = potential_per_particle.sum()
    return forces, total_potential, potential_per_particle


def no_force_fp(q: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fallenfunktion, die keine Kraft und kein Potential zurückgibt.
    """
    forces = torch.zeros_like(q)
    total_potential = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    potential_per_particle = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    return forces, total_potential, potential_per_particle

def no_pair_force_fp(q: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Funktion für die Paar-Wechselwirkung (pair_force_func), 
    die keine Kraft und kein Potential zurückgibt.
    """
    forces = torch.zeros_like(q)
    total_potential = torch.tensor(0.0, device=q.device, dtype=q.dtype)
    return forces, total_potential


def solve_harmonic_analytical(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    mass: float = 1.0, 
    **kwargs 
) -> Dict[str, torch.Tensor]:
    """
    Löst die Bewegung im harm. Oszillator exakt
    """
    device, dtype = q0.device, q0.dtype

    # 1. Kraftmatrix K = Ω² und Identitäts-/Nullmatrizen aufbauen
    K = omega_matrix @ omega_matrix
    I = torch.eye(3, device=device, dtype=dtype)
    Z = torch.zeros((3, 3), device=device, dtype=dtype)

    # 2. 6x6 Zustandsmatrix A aufbauen
    A_top = torch.cat((Z, I / mass), dim=1)
    A_bottom = torch.cat((-K, Z), dim=1)
    A = torch.cat((A_top, A_bottom), dim=0)

    # 3. Anfangszustandsvektor z0 = (q0, p0) 
    z0 = torch.cat((q0, p0), dim=1) # Form: (n_particles, 6)

    num_save_points, n_particles = t_values.shape[0], q0.shape[0]
    
    # --- Output-Arrays initialisieren ---
    q_out = torch.empty((num_save_points, n_particles, 3), dtype=dtype, device=device)
    p_out = torch.empty((num_save_points, n_particles, 3), dtype=dtype, device=device)
    kinetic_energy_out = torch.empty(num_save_points, dtype=dtype, device=device)
    potential_harmonic_out = torch.empty(num_save_points, dtype=dtype, device=device)
    
    # Paar-Potenzial ist per Definition 0 
    potential_pair_out = torch.zeros(num_save_points, dtype=dtype, device=device)

    # --- Initialisierung (t=0) ---
    q_out[0], p_out[0] = q0, p0
    kinetic_energy_out[0] = 0.5 * torch.sum(p0**2)
    _, pot_h, _ = harmonic_fp(q0, omega_matrix)
    potential_harmonic_out[0] = pot_h

    # --- Hauptschleife (t > 0) ---
    for i, t in enumerate(t_values[1:], 1):
        # Berechne den Propagator
        propagator = torch.matrix_exp(A * (t - t_values[i-1])) # Propagiert von t_i-1 zu t_i
        
        # z0 geht nicht 
    
        z_last = torch.cat((q_out[i-1], p_out[i-1]), dim=1)

        # Propagiere den Zustand
        zt = z_last @ propagator.T  # Transponieren
        
        # Spalte den Ergebnisvektor wieder in Position und Impuls auf
        q_current = zt[:, :3]
        p_current = zt[:, 3:]
        
        # Speichere q und p
        q_out[i] = q_current
        p_out[i] = p_current
        
        # Berechne und speichere Energien
        kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
        _, pot_h, _ = harmonic_fp(q_current, omega_matrix)
        potential_harmonic_out[i] = pot_h
        
    # --- Rückgabe als Dictionary ---
    return {
        "times": t_values, "positions": q_out, "momenta": p_out,
        "kinetic_energy": kinetic_energy_out,
        "potential_energy_harmonic": potential_harmonic_out,
        "potential_energy_pair": potential_pair_out
    }

def harmonic_fp(
    q: torch.Tensor,
    omega_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega_squared = omega_matrix @ omega_matrix.T
    forces = - q @ omega_squared
    potential_per_particle = 0.5 * torch.einsum('ni,ij,nj->n', q, omega_squared, q)
    total_potential = potential_per_particle.sum()
    return forces, total_potential, potential_per_particle

def run_verlet_simulation_HO(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    pair_force_func: Callable,
    pair_force_params: Dict,
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Führt eine Simulation mit dem Velocity Verlet-Algorithmus durch.

    """
    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_harmonic_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Initialisierung ---
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        f_h_current, pot_h, _ = harmonic_fp(q_current, omega_matrix)
        f_p_current, pot_p = pair_force_func(q_current, **pair_force_params)
        a_current = f_h_current + f_p_current
        potential_harmonic_out[0], potential_pair_out[0] = pot_h, pot_p

        p_half = torch.empty_like(p_current)
        q_next = torch.empty_like(q_current)
        a_next = torch.empty_like(a_current)
        p_next = torch.empty_like(p_current)

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps

            # --- INNERE SCHLEIFE ---
            for _ in range(substeps):
                p_half = p_current + 0.5 * dt * a_current
                q_next = q_current + dt * p_half

                f_h_next, pot_h_next, _ = harmonic_fp(q_next, omega_matrix)
                f_p_next, pot_p_next = pair_force_func(q_next, **pair_force_params)
                a_next = f_h_next + f_p_next

                p_next = p_half + 0.5 * dt * a_next

                # Verwende In-Place-Swapping für Puffer
                q_current, q_next = q_next, q_current
                p_current, p_next = p_next, p_current
                a_current, a_next = a_next, a_current
            # --- ENDE INNERE SCHLEIFE ---

            # Zustand am Ende in die Output-Arrays schreiben
            q_out[i], p_out[i] = q_current, p_current
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_harmonic_out[i] = pot_h_next
            potential_pair_out[i] = pot_p_next

            loop_end_time = time.perf_counter()
            eta_seconds = int((loop_end_time - loop_start_time) * (num_save_points - 1 - i))
            print(f"\rIntegration {100 * (i + 1) / num_save_points:.0f}% "
                  f"| Time: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)



        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_harmonic": potential_harmonic_out,
            "potential_energy_pair": potential_pair_out
        }



def run_DKDsplitting_simulation_HO(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    pair_force_func: Callable,
    pair_force_params: Dict,
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Simulation mit dem Strang Splitting (Drift-Kick-Drift)
    
    H = T(p) + V(q)

    """
    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        # Output-Arrays
        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_harmonic_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Initialisierung ---
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        
        # Berechne Potentiale für die Speicherung
      
        _, pot_h, _ = harmonic_fp(q_current, omega_matrix)
        _, pot_p = pair_force_func(q_current, **pair_force_params)
        potential_harmonic_out[0], potential_pair_out[0] = pot_h, pot_p


        q_half = torch.empty_like(q_current)

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps
            dt_half = 0.5 * dt # Halber Zeitschritt

            # --- INNERE SCHLEIFE (Drift-Kick-Drift) ---
            for _ in range(substeps):
                
                # 1. DRIFT (dt/2): p ist konstant
                # q_half = q_current + p_current * dt_half
                torch.add(q_current, p_current, alpha=dt_half, out=q_half)
                
                # 2. KICK (dt): q ist konstant
                
                f_h_half, pot_h_half, _ = harmonic_fp(q_half, omega_matrix)
                f_p_half, pot_p_half = pair_force_func(q_half, **pair_force_params)
                
                # p_current = p_current + (f_h_half + f_p_half) * dt
                p_current.add_(f_h_half, alpha=dt)
                p_current.add_(f_p_half, alpha=dt)
                
                # 3. DRIFT (dt/2): p ist konstant
                # q_current = q_half + p_current * dt_half
                torch.add(q_half, p_current, alpha=dt_half, out=q_current)

            # --- ENDE INNERE SCHLEIFE ---

            # Zustand am Ende in die Output-Arrays schreiben
            q_out[i], p_out[i] = q_current, p_current
            
            # Energien berechnen 
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_harmonic_out[i] = pot_h_half
            potential_pair_out[i] = pot_p_half

            loop_end_time = time.perf_counter()
            eta_seconds = int((loop_end_time - loop_start_time) * (num_save_points - 1 - i))
            print(f"\rIntegration {100 * (i + 1) / num_save_points:.0f}% "
                  f"| Time: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)

        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_harmonic": potential_harmonic_out,
            "potential_energy_pair": potential_pair_out
        }

### Dipole Trap Teil 



def run_verlet_simulation_general(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    trap_force_func: Callable,   # calculate_crossed_beam_dipole_potential
    trap_force_params: Dict,    # Parameter für die Falle
    pair_force_func: Callable,    # pair_keops_fp
    pair_force_params: Dict,    # Parameter für die WW
    precision_type: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    substeps: int = 100,
    silent: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """simulation mit Velocity Verlet """
    with torch.no_grad():
        num_save_points, n_particles = t_values.size(0), q0.size(0)

        q_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        p_out = torch.empty((num_save_points, n_particles, 3), dtype=precision_type, device=device)
        kinetic_energy_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_trap_out = torch.empty(num_save_points, dtype=precision_type, device=device)
        potential_pair_out = torch.empty(num_save_points, dtype=precision_type, device=device)

        q_current, p_current = q0.to(device, precision_type), p0.to(device, precision_type)

        # --- Initialisierung ---
        q_out[0], p_out[0] = q_current, p_current
        kinetic_energy_out[0] = 0.5 * torch.sum(p_current**2)
        f_trap_current, pot_trap, _ = trap_force_func(q_current, **trap_force_params)
        f_pair_current, pot_pair = pair_force_func(q_current, **pair_force_params)
        a_current = f_trap_current + f_pair_current
        potential_trap_out[0], potential_pair_out[0] = pot_trap, pot_pair

        p_half = torch.empty_like(p_current)
        q_next = torch.empty_like(q_current)
        a_next = torch.empty_like(a_current)
        p_next = torch.empty_like(p_current)

        # --- Hauptschleife ---
        for i in range(1, num_save_points):
            loop_start_time = time.perf_counter()
            Dt = t_values[i] - t_values[i - 1]
            dt = Dt / substeps

            for _ in range(substeps):
                p_half = p_current + 0.5 * dt * a_current
                q_next = q_current + dt * p_half

                f_trap_next, pot_trap_next, _ = trap_force_func(q_next, **trap_force_params)
                f_pair_next, pot_pair_next = pair_force_func(q_next, **pair_force_params)
                a_next = f_trap_next + f_pair_next
                p_next = p_half + 0.5 * dt * a_next

                q_current, q_next = q_next, q_current
                p_current, p_next = p_next, p_current
                a_current, a_next = a_next, a_current

            q_out[i], p_out[i] = q_current, p_current
            kinetic_energy_out[i] = 0.5 * torch.sum(p_current**2)
            potential_trap_out[i] = pot_trap_next
            potential_pair_out[i] = pot_pair_next

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
