import torch
import math
from typing import Dict, Tuple, Callable

# --- GPU-MCMC Sampling für zwei Temperatur-Gruppen ---
def generate_positions_mcmc_two_temp(
    n_particles_groups: Tuple[int, int],
    temp_k_groups: Tuple[float, float],
    trap_force_func: Callable,
    pair_potential_func: Callable,
    trap_params: Dict,
    pair_params: Dict,
    temp_ref_k: float,
    device: torch.device = torch.device('cuda'),
    precision: torch.dtype = torch.float32,
    n_steps: int = 5000,
    step_size: float = 0.1
) -> torch.Tensor:
    """
    GPU-MCMC Sampling für zwei Temperaturniveaus unter Berücksichtigung von 
    Trap-Potential + paarweiser Repulsion (softcore). 

    Returns
    -------
    q_all : torch.Tensor
        Positionen aller Teilchen beider Gruppen (N_total, 3)
    """
    all_positions = []

    for n_particles, temp_k in zip(n_particles_groups, temp_k_groups):
        if n_particles == 0:
            continue

        temp_ratio = temp_k / temp_ref_k

        # --- 1. Startpositionen zufällig ---
        q = 2.0 * torch.rand(n_particles, 3, device=device, dtype=precision) - 1.0

        # --- 2. MCMC Schritte ---
        for step in range(n_steps):
            dq = torch.randn_like(q) * step_size
            q_new = q + dq

            # Trap-Potential
            _, pot_trap_old, _ = trap_force_func(q, **trap_params)
            _, pot_trap_new, _ = trap_force_func(q_new, **trap_params)

            # Paarpotential
            _, pot_pair_old = pair_potential_func(q, **pair_params)
            _, pot_pair_new = pair_potential_func(q_new, **pair_params)

            # Gesamtenergieänderung
            delta_U = (pot_trap_new + pot_pair_new) - (pot_trap_old + pot_pair_old)

            # Akzeptanzwahrscheinlichkeit (Boltzmann)
            p_accept = torch.exp(-delta_U / temp_ratio)
            u = torch.rand(1, device=device, dtype=precision)
            if u < p_accept:
                q = q_new  # akzeptieren

        all_positions.append(q)

    q_all = torch.cat(all_positions, dim=0)
    return q_all


# --- Angepasste Zwei-Temperaturen-Initialisierung ---
def prepare_two_temp_simulation_mcmc(
    n_particles_groups: Tuple[int, int],
    temp_k_groups: Tuple[float, float],
    omega_phys_hz: Tuple[float, float, float],  # Für Bounding Box & Skalierung
    trap_force_func: Callable,
    pair_potential_func: Callable,
    trap_params: Dict,
    pair_params: Dict,
    t_end_s: float,
    dt_s: float,
    r0_phys: float = None,
    C_phys: float = None,
    precision: torch.dtype = torch.float32,
    device: torch.device = torch.device('cuda'),
    n_mcmc_steps: int = 5000,
    step_size: float = 0.1
) -> Dict:
    """
    Bereitet eine Zwei-Temperaturen-Simulation vor:
    - Positionen werden mittels MCMC erzeugt unter Berücksichtigung von Trap-Potential
      und paarweiser Repulsion.
    - Impulse werden als Gauß-Vektoren entsprechend der Temperatur verteilt.
    """
    # --- 1. Skalierung basierend auf Gruppe 1 ---
    mass_kg = 86.909 * 1.66054e-27  # Masse in kg
    temp_ref_k = temp_k_groups[0]

    omega_phys_rad_s = 2 * math.pi * torch.tensor(omega_phys_hz, dtype=precision, device=device)
    m0, E0 = mass_kg, 1.380649e-23 * temp_ref_k  # kB*T_ref
    omega_char_rad_s = omega_phys_rad_s[0]
    L0 = torch.sqrt(E0 / (m0 * omega_char_rad_s**2))
    T0 = 1.0 / omega_char_rad_s

    # --- 2. Positionen mittels MCMC Sampling ---
    q0_dimless = generate_positions_mcmc_two_temp(
        n_particles_groups=n_particles_groups,
        temp_k_groups=temp_k_groups,
        trap_force_func=trap_force_func,
        pair_potential_func=pair_potential_func,
        trap_params=trap_params,
        pair_params=pair_params,
        temp_ref_k=temp_ref_k,
        device=device,
        precision=precision,
        n_steps=n_mcmc_steps,
        step_size=step_size
    )

    # --- 3. Impulse (Gauß) entsprechend Temperatur ---
    p_list = []
    for n_particles, temp_k in zip(n_particles_groups, temp_k_groups):
        if n_particles == 0:
            continue
        temp_ratio = temp_k / temp_ref_k
        sigma_p_dimless = math.sqrt(temp_ratio)
        p_group = torch.randn(n_particles, 3, dtype=precision, device=device) * sigma_p_dimless
        p_group -= torch.mean(p_group, dim=0, keepdim=True)  # zentriere Impuls der Gruppe
        p_list.append(p_group)

    p0_dimless = torch.cat(p_list, dim=0)
    p0_dimless -= torch.mean(p0_dimless, dim=0, keepdim=True)  # Gesamtimpuls zentrieren

    # --- 4. Zeitschritte & Dimensionslose Frequenzen ---
    t_values = torch.arange(0, t_end_s / T0, dt_s / T0, dtype=precision, device=device)
    omega_dimless = omega_phys_rad_s / omega_char_rad_s
    omega_matrix_dimless = torch.diag(omega_dimless)

    # --- 5. Paar-Parameter in dimensionsloser Form ---
    pair_params_dimless = {'r': 0.0, 'c': 0.0}
    if r0_phys is not None and C_phys is not None:
        pair_r_dimless = r0_phys / L0
        pair_c_dimless = C_phys / E0
        pair_params_dimless['r'] = float(pair_r_dimless)
        pair_params_dimless['c'] = float(pair_c_dimless)

    return {
        "t_values": t_values,
        "q0": q0_dimless,
        "p0": p0_dimless,
        "omega_matrix_dimless": omega_matrix_dimless,
        "pair_force_params": pair_params_dimless,
        "precision_type": precision,
        "device": device,
        "L0_m": L0,
        "T0_s": T0,
        "E0_J": E0
    }
