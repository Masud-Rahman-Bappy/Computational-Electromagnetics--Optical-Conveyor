"""Vector Bessel-beam fields at an air-liquid interface.

The field expressions follow Eq. (1) and the refracted-beam construction in
Rahman et al., Annalen der Physik 527, 777-793 (2015),
DOI 10.1002/andp.201500266. Arrays use NumPy's ``(y, x)`` image convention.

The complex phase convention is ``exp(i*(beta*z + m*phi - omega*t))``.  It is
the convention used by the paper to derive the interference phase in Eq. (5b).
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv


MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12


def bessel_beam_gas_liquid_force(
    incident_angle, m, Ce, Ch, xrange, yrange, loop, z, t, Parx,
    Delw, Delb, meu_g_r, epshilon_g_r, meu_l_r, epshilon_l_r,
    lamda, rngg, direction,
):
    """Calculate the gas/liquid Bessel-beam electric-field components.

    ``loop`` selects beam 1 or beam 2. A value of 0 is also accepted for the
    first beam, allowing convenient zero-based calls.
    """
    angles_deg = np.asarray(incident_angle, dtype=float)
    rngg = np.asarray(rngg, dtype=float)
    x0 = np.asarray(xrange, dtype=float)
    y0 = -np.asarray(yrange, dtype=float)
    idx = int(loop - 1) if loop in (1, 2) else int(loop)

    mu_g, eps_g = meu_g_r * MU0, epshilon_g_r * EPS0
    mu_l, eps_l = meu_l_r * MU0, epshilon_l_r * EPS0
    c0 = 1.0 / np.sqrt(mu_g * eps_g)
    omega0 = 2.0 * np.pi * c0 / lamda
    k0 = 2.0 * np.pi / lamda
    parax = np.deg2rad(Parx)
    q0, beta0 = k0 * np.sin(parax), k0 * np.cos(parax)
    wp_theory = omega0 * Delw / (2.0 * abs(m))

    incidence = np.deg2rad(angles_deg)
    g_ul = np.sqrt((mu_l * eps_l) / (mu_g * eps_g))
    refraction = np.arcsin(np.sin(incidence) / g_ul)
    save_i, save_r = incidence[idx], refraction[idx]

    delta_r = refraction[0] - direction * refraction[1]
    ratio = (rngg[1] / rngg[0]) * (
        (np.cos(refraction[1]) / np.cos(incidence[1]))
        / (np.cos(refraction[0]) / np.cos(incidence[0]))
    )
    theta_one = np.arctan2(np.sin(delta_r), ratio + np.cos(delta_r))
    theta_two = delta_r - theta_one

    omega, beta, q = omega0, beta0, q0
    theta_nat = theta_one if idx == 0 else -theta_two
    delta_beta = (
        beta0 * np.cos(theta_one) - (beta0 - beta0 * Delb) * np.cos(theta_two)
    ) * g_ul
    if idx != 0:
        omega = omega0 * (1.0 - Delw)
        beta = beta0 * (1.0 - Delb)
        lamda = c0 / (omega / (2.0 * np.pi))
        k = 2.0 * np.pi / lamda
        # Keep q real for a propagating radial component.  Use an imaginary q
        # only when the radial component is evanescent.
        q_squared = k * k - beta * beta
        q = np.sqrt(q_squared) if q_squared >= 0.0 else 1j * np.sqrt(-q_squared)

    gamma = np.sqrt(
        1.0 + (1.0 - 1.0 / 1.33**2) * np.tan(incidence[idx]) ** 2
    )
    sin_theta_nat = np.sin(theta_nat)
    max_range = (
        np.inf if np.isclose(sin_theta_nat, 0.0)
        else 2.0 * rngg[idx] * gamma / sin_theta_nat
    )
    # np.rad2deg does not accept complex values.  Direct multiplication works
    # for both real propagation angles and complex evanescent angles.
    cone_angle = np.real_if_close(np.arctan(q / beta) * (180.0 / np.pi))

    # Local cylindrical coordinates about the refracted beam axis.  This is
    # the (rho_l, phi_l, z_l) frame used for the transmitted fields in Eq. (5a).
    X, Y = np.meshgrid(x0, y0, indexing="xy")
    axial = z * np.sin(theta_nat) + X * np.cos(theta_nat)
    Z = z * np.cos(theta_nat) - X * np.sin(theta_nat)
    rho = np.hypot(axial, Y)
    phi = np.arctan2(Y, axial)
    rho_safe = np.where(rho == 0.0, 1e-25, rho)

    Jm = jv(m, q * rho)
    Jm_der = 0.5 * (jv(m - 1, q * rho) - jv(m + 1, q * rho))
    gE_rho0 = -(m * omega * mu_g / q**2) * Ch * Jm / rho_safe + 1j * (beta / q) * Ce * Jm_der
    gE_phi0 = -(m * beta / q**2) * Ce * Jm / rho_safe - 1j * (omega * mu_g / q) * Ch * Jm_der
    gE_z0 = Ce * Jm
    gEx = gE_rho0 * np.cos(phi) - gE_phi0 * np.sin(phi)
    gEy = gE_rho0 * np.sin(phi) + gE_phi0 * np.cos(phi)
    phase_g = np.exp(1j * (beta * Z - omega * t + m * phi))
    gE_rho, gE_phi, gE_z = gE_rho0 * phase_g, gE_phi0 * phase_g, gE_z0 * phase_g

    eta_g, eta_l = np.sqrt(mu_g / eps_g), np.sqrt(mu_l / eps_l)
    eps_rg, eps_rl = 1.0, eps_l / eps_g
    tx = np.array([
        [1, 0, 0, -1, 0, 0], [1 / eta_l, 0, 0, -1 / eta_g, 0, 0],
        [0, 1, 0, 0, -1, 0], [0, 1 / eta_l, 0, 0, -1 / eta_g, 0],
        [0, 0, eps_rl, 0, 0, -eps_rg],
        [0, 0, eps_rl / eta_l, 0, 0, -eps_rg / eta_g],
    ], dtype=float)
    inv_tx = np.linalg.inv(tx)
    lEx0 = (inv_tx[0, 0] + inv_tx[0, 1] / eta_g) * gEx
    lEy0 = (inv_tx[1, 2] + inv_tx[1, 3] / eta_g) * gEy
    lEz0 = (inv_tx[2, 4] + inv_tx[2, 5] / eta_g) * eps_rg * gE_z0
    rEx0 = (inv_tx[3, 0] + inv_tx[3, 1] / eta_g) * gEx
    rEy0 = (inv_tx[4, 2] + inv_tx[4, 3] / eta_g) * gEy
    rEz0 = (inv_tx[5, 4] + inv_tx[5, 5] / eta_g) * eps_rg * gE_z0

    rEx = rEx0 * np.cos(np.pi - save_i) + lEz0 * np.sin(np.pi - save_i)
    rEz = -rEx0 * np.sin(np.pi - save_i) + lEz0 * np.cos(np.pi - save_i)
    rE_rho0 = rEx * np.cos(phi) + rEy0 * np.sin(phi)
    rE_phi0 = -rEx * np.sin(phi) + rEy0 * np.cos(phi)
    phase_r = np.exp(1j * (beta * Z - omega * t + m * phi))

    # Refraction maps circular incident-beam contours into the elliptical
    # transverse coordinates illustrated in Fig. 1(d) of the paper.
    ratio_ab = np.sqrt(1.0 + (1.0 - 1.0 / g_ul**2) * np.tan(save_i) ** 2)
    angle_t = np.mod(np.arctan2(-X * ratio_ab**2, Y), np.pi)
    angle_t = np.where((X == 0.0) & (Y == 0.0), 0.0, angle_t)
    lE_rho0 = np.cos(phi) * gEx + np.sin(phi) * gEy
    lE_tee0 = np.cos(angle_t) * gEx + np.sin(angle_t) * gEy
    lE_zee0 = gE_z0
    beta_l = beta * g_ul
    phase_l = np.exp(1j * (beta_l * Z - omega * t + m * phi))

    return {
        "x": x0 / np.cos(theta_nat), "y": y0,
        "gE_rho": gE_rho, "gE_phi": gE_phi, "gE_z": gE_z,
        "rErho": rE_rho0 * phase_r, "rEphi": rE_phi0 * phase_r,
        "rEzee": rEz0 * phase_r,
        "xl": x0 / np.cos(theta_nat) * gamma, "yl": y0,
        "lE_rho": lE_rho0 * phase_l, "lE_tee": lE_tee0 * phase_l,
        "lE_zee": lE_zee0 * phase_l, "save_r": save_r,
        "VzTheory": -(omega0 * Delw) / delta_beta,
        "WpTheory": wp_theory, "coneAng": cone_angle,
        "beta": beta_l, "w": omega, "MaxRng": max_range,
    }


# Descriptive public alias for callers that prefer the paper's beam name.
BesselBeamGasLiquidForce = bessel_beam_gas_liquid_force
