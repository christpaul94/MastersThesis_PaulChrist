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
