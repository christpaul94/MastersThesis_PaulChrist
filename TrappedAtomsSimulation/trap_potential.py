import torch
import math
from typing import Dict, Tuple

import torch
import math
from typing import Tuple

def calculate_single_beam_X_AXIS(
    positions: torch.Tensor,
    P: float, w0: float, s0: float, 
    omega_0: float, Gamma: float, Delta: float, 
    L0: float,  
    E0: float    
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Berechnung für einen Gauß-Strahl, der entlang der X-ACHSE propagiert.
    
    """
    with torch.no_grad():
        # --- 1. Skalierung & SI-Einheiten ---
        
        hbar = 1.054571817e-34
        c_light = 299_792_458.0
        
        r = positions * L0 # Verwende L0

        # --- 2. Geometrie (Hartkodiert für X-Achse) ---
        s = r[:, 0]  # s = x
        rho2 = r[:, 1].pow(2) + r[:, 2].pow(2) # rho^2 = y^2 + z^2

        # --- 3. Intensität & Potential  ---
        w_s = w0 * torch.sqrt(1 + (s / s0).pow(2))
        intensity = (2 * P) / (math.pi * w_s.pow(2)) * torch.exp(-2 * rho2 / w_s.pow(2))
        prefactor = (3 * math.pi * c_light**2) / (2 * omega_0**3) * (Gamma / Delta)
        potential_SI = prefactor * intensity
        
        potential_dimless = potential_SI / E0 # Verwende E0
        total_potential = torch.sum(potential_dimless)

        # --- 4. Kraftberechnung  ---
        dw_ds = w0 * (s / s0**2) / torch.sqrt(1 + (s / s0).pow(2))
        dU_ds = potential_SI * (4 * rho2 / w_s.pow(3) - 2 / w_s) * dw_ds
        dU_drho2 = -2 * potential_SI / w_s.pow(2)

        grads_U = torch.empty_like(r)
        
        # ∇U = (∂U/∂s) * ∇s + (∂U/∂ρ²) * ∇(ρ²)
        # ∇s = (1, 0, 0)
        # ∇(rho^2) = (0, 2y, 2z)
        
        grads_U[:, 0] = dU_ds                # dU/dx = dU/ds * 1
        grads_U[:, 1] = dU_drho2 * 2 * r[:, 1] # dU/dy = dU/d(rho^2) * 2y
        grads_U[:, 2] = dU_drho2 * 2 * r[:, 2] # dU/dz = dU/d(rho^2) * 2z

        
        forces_dimless = -grads_U * L0 / E0 # Verwende L0 und E0
        
        return forces_dimless, total_potential.item(), potential_dimless


import torch
import math
from typing import Tuple

def calculate_single_beam_Y_AXIS(
    positions: torch.Tensor,
    P: float, w0: float, s0: float, 
    omega_0: float, Gamma: float, Delta: float, 
    L0: float,  # Parameter (statt a_ho)
    E0: float   # Parameter (statt E_ho)
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Berechnung für einen Gauß-Strahl, der entlang der Y-ACHSE propagiert.
    
    """
    with torch.no_grad():
        # --- 1. Skalierung & SI-Einheiten ---
        hbar = 1.054571817e-34
        c_light = 299_792_458.0
        
        r = positions * L0 # Verwende L0

        # --- 2. Geometrie   ---
        s = r[:, 1]  # s = y
        rho2 = r[:, 0].pow(2) + r[:, 2].pow(2) # rho^2 = x^2 + z^2

        # --- 3. Intensität & Potential ---
        w_s = w0 * torch.sqrt(1 + (s / s0).pow(2))
        intensity = (2 * P) / (math.pi * w_s.pow(2)) * torch.exp(-2 * rho2 / w_s.pow(2))
        prefactor = (3 * math.pi * c_light**2) / (2 * omega_0**3) * (Gamma / Delta)
        potential_SI = prefactor * intensity
        
        potential_dimless = potential_SI / E0 # Verwende E0
        total_potential = torch.sum(potential_dimless)

        # --- 4. Kraftberechnung   ---
        dw_ds = w0 * (s / s0**2) / torch.sqrt(1 + (s / s0).pow(2))
        dU_ds = potential_SI * (4 * rho2 / w_s.pow(3) - 2 / w_s) * dw_ds
        dU_drho2 = -2 * potential_SI / w_s.pow(2)

        grads_U = torch.empty_like(r)
        
        # ∇U = (∂U/∂s) * ∇s + (∂U/∂ρ²) * ∇(ρ²)
        # ∇s = (0, 1, 0)
        # ∇(rho^2) = (2x, 0, 2z)
        
        grads_U[:, 0] = dU_drho2 * 2 * r[:, 0] # dU/dx = dU/d(rho^2) * 2x
        grads_U[:, 1] = dU_ds                # dU/dy = dU/ds * 1
        grads_U[:, 2] = dU_drho2 * 2 * r[:, 2] # dU/dz = dU/d(rho^2) * 2z

        # --- 5. Rückgabe ---
        forces_dimless = -grads_U * L0 / E0 # Verwende L0 und E0
        
        return forces_dimless, total_potential.item(), potential_dimless

import torch
import math
from typing import Tuple

# --- Hier müssen die beiden achsenspezifischen Funktionen definiert sein ---
# def calculate_single_beam_X_AXIS(...): ...
# def calculate_single_beam_Y_AXIS(...): ...
# ---

def calculate_crossed_beam_dipole_potential(
    positions: torch.Tensor,
    P_x: float,         # Leistung des Strahls in x-Richtung
    P_y: float,         # Leistung des Strahls in y-Richtung
    w0_x: float,        # Strahl Taille für x-Strahl
    w0_y: float,        # Strahl Taille für y-Strahl
    s0_x: float,        # Rayleigh-Länge für x-Strahl
    s0_y: float,        # Rayleigh-Länge für y-Strahl
    omega_0: float,
    Gamma: float,
    Delta: float,
    L0: float,          # NEU: Als Parameter übergeben
    E0: float,          # NEU: Als Parameter übergeben
    **kwargs            # Nimmt alle weiteren Argumente auf
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Berechnet das Potential und die Kräfte für eine gekreuzte Dipolfalle
 
    """
 

    # --- 2. Berechnung für den Strahl in x-Richtung ---
    forces_x, pot_total_x, pot_particles_x = calculate_single_beam_X_AXIS(
        positions=positions, 
        P=P_x, w0=w0_x, s0=s0_x, 
        omega_0=omega_0, Gamma=Gamma, Delta=Delta,
        L0=L0, E0=E0  # Übergabe der Skalierungsfaktoren
    )

    # --- 3. Berechnung für den Strahl in y-Richtung ---
    forces_y, pot_total_y, pot_particles_y = calculate_single_beam_Y_AXIS(
        positions=positions, 
        P=P_y, w0=w0_y, s0=s0_y, 
        omega_0=omega_0, Gamma=Gamma, Delta=Delta,
        L0=L0, E0=E0  # Übergabe der Skalierungsfaktoren
    )

    # --- 4. Überlagerung der Ergebnisse ---
    total_forces = forces_x + forces_y
    total_potential = pot_total_x + pot_total_y
    total_potential_per_particle = pot_particles_x + pot_particles_y

    return total_forces, total_potential, total_potential_per_particle
