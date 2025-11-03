def initialize_momenta_thermal(
    n_particles: int, temperature: float, mass: float, kB: float, precision_type: torch.dtype
) -> torch.Tensor:
    """Initialisiert Impulse aus einer Maxwell-Boltzmann-Verteilung."""
    sigma_p = (mass * kB * temperature)**0.5
    momenta = torch.randn(n_particles, 3, dtype=precision_type) * sigma_p
    if n_particles > 1:
        momenta -= torch.mean(momenta, dim=0) # Gesamtimpuls auf Null setzen
    return momenta


