def solve_harmonic_analytical(
    t_values: torch.Tensor,
    q0: torch.Tensor,
    p0: torch.Tensor,
    omega_matrix: torch.Tensor,
    mass: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:

    device, dtype = q0.device, q0.dtype

    # 1. Kraftmatrix K = Ω² und Identitäts-/Nullmatrizen aufbauen
    K = omega_matrix @ omega_matrix
    I = torch.eye(3, device=device, dtype=dtype)
    Z = torch.zeros((3, 3), device=device, dtype=dtype)

    # 2. x6 Zustandsmatrix A f
    A_top = torch.cat((Z, I / mass), dim=1)
    A_bottom = torch.cat((-K, Z), dim=1)
    A = torch.cat((A_top, A_bottom), dim=0)

    # 3. Anfangszustandsvektor z0 = (q0, p0) für alle Teilchen vorbereiten
    z0 = torch.cat((q0, p0), dim=1) # Form: (n_particles, 6)

    num_save_points, n_particles = t_values.shape[0], q0.shape[0]
    q_out = torch.empty((num_save_points, n_particles, 3), dtype=dtype, device=device)
    p_out = torch.empty((num_save_points, n_particles, 3), dtype=dtype, device=device)
    q_out[0], p_out[0] = q0, p0

    for i, t in enumerate(t_values[1:], 1):
        #
        propagator = torch.matrix_exp(A * t)

        zt = z0 @ propagator.T  # Transponieren,

        # Spalte den Ergebnisvektor wieder in Position und Impuls auf
        q_out[i] = zt[:, :3]
        p_out[i] = zt[:, 3:]

    return q_out, p_out


def run_velocity_verlet_simulation_HO(
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
                  f"| ETA: {eta_seconds // 60} min {eta_seconds % 60} s", end='', flush=True)


            print("\nIntegration abgeschlossen.") # Fügt einen Zeilenumbruch hinzu

        return {
            "times": t_values, "positions": q_out, "momenta": p_out,
            "kinetic_energy": kinetic_energy_out,
            "potential_energy_harmonic": potential_harmonic_out,
            "potential_energy_pair": potential_pair_out
        }
