import torch
from typing import Dict, Tuple, Callable

def plot_simple_momentum_histogram(
    p_tensor_dimless: torch.Tensor,
    bins: int = 75,
    title: str = "Momentum Distribution"
):

    p_magnitudes = torch.sqrt(torch.sum(p_tensor_dimless**2, dim=1))

    p_magnitudes_np = p_magnitudes.cpu().numpy()

    plt.figure(figsize=(10, 6))

    plt.hist(p_magnitudes_np, bins=bins, density=True, alpha=0.8, label='Simulated Distribution')

    plt.title(title, fontsize=16)
    plt.xlabel("Dimensionsloser Impulsbetrag |p|")
    plt.ylabel("Wahrscheinlichkeitsdichte")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()

def calculate_group_temperatures(
    momenta: torch.Tensor,
    n_groups: Tuple[int, int],
    T_ref: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    n1 = n_groups[0]
    p_group1 = momenta[:, :n1, :]
    p_group2 = momenta[:, n1:, :]
    K_dimless_group1 = 0.5 * torch.sum(p_group1**2, dim=(1, 2))
    K_dimless_group2 = 0.5 * torch.sum(p_group2**2, dim=(1, 2))
    T_group1 = T_ref * (2.0 / (3.0 * n_groups[0])) * K_dimless_group1
    T_group2 = T_ref * (2.0 / (3.0 * n_groups[1])) * K_dimless_group2
    return T_group1, T_group2

def plot_thermalization(
    results: Dict,
    n_groups: Tuple[int, int],
    temp_groups: Tuple[float, float],
    T0_s: torch.Tensor # It receives a tensor
):
    T_ref = temp_groups[0]
    T_group1, T_group2 = calculate_group_temperatures(results['momenta'], n_groups, T_ref)
    t_phys_ms = results['times'].cpu().numpy() * T0_s.item() * 1000
    T_eq = (n_groups[0] * temp_groups[0] + n_groups[1] * temp_groups[1]) / sum(n_groups)

    plt.figure(figsize=(12, 7))
    plt.plot(t_phys_ms, T_group1.cpu().numpy() * 1e6, label=f"Gruppe 1 ({n_groups[0]} Teilchen)", color='blue')
    plt.plot(t_phys_ms, T_group2.cpu().numpy() * 1e6, label=f"Gruppe 2 ({n_groups[1]} Teilchen)", color='red')
    plt.axhline(y=T_eq * 1e6, color='black', linestyle='--', label=f"T_eq = {T_eq*1e6:.2f} µK")
    plt.title("Thermalisierung der Teilchengruppen", fontsize=16)
    plt.xlabel("Zeit (ms)")
    plt.ylabel("Temperatur (µK)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


def plot_energy_and_error(t, kinetic_energy, potential_pl, potential_lj):
    """
    Plots:
    1. Energy contributions (linear scale)
    2. Energy contributions (log scale)
    3. Relative total energy deviation (log scale)
    All in subplots.
    """
    energy_contributions = [kinetic_energy]
    if potential_pl is not None:
        energy_contributions.append(potential_pl)

    if potential_lj is not None:
        energy_contributions.append(potential_lj)

    # Gesamtenergie und Fehler
    E_total = sum(energy_contributions)
    E0 = E_total[0]
    delta_E_rel = torch.abs(E_total - E0) / torch.abs(E0)

    t_np = t.cpu().numpy()

    # Subplots (3 Zeilen, 1 Spalte)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # 1. Lineare Skala
    axes[0].plot(t_np, kinetic_energy.cpu().numpy(), label='Kinetic Energy')
    if potential_pl is not None:
        axes[0].plot(t_np, potential_pl.cpu().numpy(), label='Potential (Power Law)')

    if potential_lj is not None:
        axes[0].plot(t_np, potential_lj.cpu().numpy(), label='Potential (Lennard-Jones)')
    axes[0].plot(t_np, E_total.cpu().numpy(), '--', color='r', label='Total Energy')
    axes[0].set_title('Energy Contributions (Linear Scale)')
    axes[0].set_ylabel(r'$\bar{E} = E / E_0$')
    #axes[0].set_ylim(0,)
    axes[0].legend()
    axes[0].grid()

    # 2. Logarithmische Skala
    axes[1].plot(t_np, torch.abs(kinetic_energy).cpu().numpy(), label='Kinetic Energy')
    if potential_pl is not None:
        axes[1].plot(t_np, torch.abs(potential_pl).cpu().numpy(), label='Potential (Power Law)')

    if potential_lj is not None:
        axes[1].plot(t_np, torch.abs(potential_lj).cpu().numpy(), label='Potential (Lennard-Jones)')
    axes[1].plot(t_np, torch.abs(E_total).cpu().numpy(), '--', color='r', label='Total Energy')
    axes[1].set_yscale('log')
    axes[1].set_title('Energy Contributions (Logarithmic Scale)')
    axes[1].set_ylabel(r'$\log|\bar{E}|$')
    axes[1].legend()
    axes[1].grid()

    # 3. Relative Energieabweichung
    axes[2].plot(t_np, delta_E_rel.cpu().numpy(), label='Relative Energy Deviation')
    axes[2].axhline(y=delta_E_rel.max().cpu(), linestyle=':', color='b', label=f'Max Error = {delta_E_rel.max():.2e}')
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-18, 1)
    axes[2].set_title('Energy Conservation Over Time')
    axes[2].set_xlabel(r'Dimensionless Time $\tau = \omega_{\mathrm{ref}} t$')
    axes[2].set_ylabel(r'$\Delta E / E_0$')
    axes[2].legend()
    axes[2].grid()

    # Layout anpassen
    plt.tight_layout()
    plt.show()

    return delta_E_rel.max()
