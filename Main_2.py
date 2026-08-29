"""15 kHz structured-Bessel-beam interference and particle visualization.

The physical model follows M. M. Rahman et al., "Tractor beam for fully
immersed multiple objects: Long distance pulling, trapping, and rotation with
a single optical set-up," Annalen der Physik 527, 777-793 (2015),
DOI 10.1002/andp.201500266.

Two Bessel beams with reverse orders (+2, -2) create four azimuthal trapping
regions. Their unequal angular frequencies and longitudinal wave numbers move
the interference landscape in the azimuthal and axial directions, respectively.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes

from BesselBeamGasLiquidForceFunc import bessel_beam_gas_liquid_force
from OpticalForceAnalysisFunc import pattern_velocities


def run_15khz_simulation(show=True):
    """Calculate the two-beam fields and optionally render the 3-D result."""
    # Beam parameters used for the 15 kHz moving interference landscape.
    Ce, Ch = 1.5e6, 0.0
    wavelength = 1.0e-6
    frequency_offset_hz = 15_000.0
    delta_omega_ratio = frequency_offset_hz / (3.0e8 / wavelength)
    delta_beta_ratio = 0.16090036886470404
    paraxial_angle_deg = 40.0
    orders = np.array([+2, -2])
    incident_angles_deg = np.array([45.0, 45.0])

    # The non-coaxial geometry inputs reduce to a coaxial pair for equal
    # incidence angles. They are retained because the field function also
    # supports the non-coaxial construction discussed with Fig. 1(c).
    beam_ranges = np.array([5.403245142485066e-7, 4.802389364443375e-7])
    beam_direction = 1

    # Particle and medium parameters used by the paper's dipole model.
    particle = dict(
        radius=100e-9,
        density=2650.0,
        permittivity=3.90 * 8.854187817e-12,
    )
    medium = dict(
        density=1000.0,
        permittivity=1.7689 * 8.854187817e-12,
        viscosity=1.002e-3,
    )
    mu_g_r, eps_g_r = 1.0, 1.0
    mu_l_r, eps_l_r = 1.0, 1.7689

    # 101 x 101 transverse points and 101 longitudinal planes.
    xy = np.linspace(-1.0e-6, 1.0e-6, 101)
    zl = np.linspace(-3.0e-6, 3.0e-6, 101)
    time_s = 0.0

    intensity_liquid = np.empty((zl.size, xy.size, xy.size), dtype=float)
    intensity_gas = np.empty_like(intensity_liquid)
    beta_values = np.empty(2)
    omega_values = np.empty(2)
    xl = yl = None
    vz_theory = wp_theory = None

    for iz, z_value in enumerate(zl):
        total = {
            key: 0.0 for key in
            ("gE_rho", "gE_phi", "gE_z", "lE_rho", "lE_tee", "lE_zee")
        }
        for beam_number, order in enumerate(orders, start=1):
            field = bessel_beam_gas_liquid_force(
                incident_angles_deg, order, Ce, Ch, xy, xy, beam_number,
                z_value, time_s, paraxial_angle_deg, delta_omega_ratio,
                delta_beta_ratio, mu_g_r, eps_g_r, mu_l_r, eps_l_r,
                wavelength, beam_ranges, beam_direction,
            )
            xl, yl = field["xl"], field["yl"]
            for key in total:
                total[key] = total[key] + field[key]
            beta_values[beam_number - 1] = field["beta"]
            omega_values[beam_number - 1] = field["w"]
            if beam_number == 1:
                vz_theory = field["VzTheory"]
                wp_theory = field["WpTheory"]

        # Paper Eq. (5a): coherent field superposition.
        # Paper Eq. (5b): the interference intensity contains axial, azimuthal,
        # and temporal phase differences between the two beams.
        liquid_raw = sum(
            np.abs(total[key]) ** 2
            for key in ("lE_rho", "lE_tee", "lE_zee")
        )
        gas_raw = sum(
            np.abs(total[key]) ** 2
            for key in ("gE_rho", "gE_phi", "gE_z")
        )
        intensity_liquid[iz] = liquid_raw / liquid_raw.max()
        intensity_gas[iz] = gas_raw

    center_index = int(np.argmin(np.abs(zl)))
    center_intensity = intensity_liquid[center_index]
    candidates = np.argwhere(center_intensity > 0.999)
    if candidates.size:
        iy_peak, ix_peak = candidates[min(3, len(candidates) - 1)]
    else:
        iy_peak, ix_peak = np.unravel_index(
            np.argmax(center_intensity), center_intensity.shape
        )
    particle_x, particle_y = float(xl[ix_peak]), float(yl[iy_peak])

    rates = pattern_velocities(omega_values, beta_values, orders)
    result = {
        "IntL": intensity_liquid,
        "IntG": intensity_gas,
        "xl": xl,
        "yl": yl,
        "zl": zl,
        "particle_position": (particle_x, particle_y, 0.0),
        "VzTheory": vz_theory,
        "WpTheory": wp_theory,
        "BETAvalue": beta_values,
        "OMEGAvalue": omega_values,
        "paper_pattern_rates": rates,
        "particle": particle,
        "medium": medium,
    }

    if show:
        plot_15khz_volume(result)
    return result


def _paper_colorscale():
    return [
        [0.00, "#242080"], [0.30, "#4154b8"],
        [0.52, "#7ed9ee"], [0.68, "#efffff"],
        [0.80, "#fff129"], [0.92, "#ff2517"],
        [1.00, "#8a0000"],
    ]


def plot_15khz_volume(
    result,
    isovalue=0.99,
    html_name="ModasserAliAkhond15kHzADP_interactive.html",
    png_name="ModasserAliAkhond15kHzADP_high_quality.png",
):
    """Render boundary, center, oblique slices, helical maxima, and particle."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Install the WebGL renderer with: python -m pip install plotly kaleido"
        ) from exc

    volume = result["IntL"]
    xl, yl, zl = result["xl"], result["yl"], result["zl"]
    particle_x, particle_y, particle_z = result["particle_position"]
    colorscale = _paper_colorscale()
    background = "#b8e9f7"
    fig = go.Figure()

    def add_slice(x, y, z, values, opacity=0.78):
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=values,
            colorscale=colorscale, cmin=0.0, cmax=1.0,
            showscale=False, opacity=opacity,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
            hovertemplate="I/Imax=%{surfacecolor:.3f}<extra></extra>",
        ))

    X_xy, Y_xy = np.meshgrid(xl, yl, indexing="xy")
    Y_yz, Z_yz = np.meshgrid(yl, zl, indexing="xy")
    X_xz, Z_xz = np.meshgrid(xl, zl, indexing="xy")

    # Two side boundaries and the lower/upper longitudinal boundaries.
    add_slice(np.full_like(Y_yz, xl[0]), Y_yz, Z_yz, volume[:, :, 0])
    add_slice(X_xz, np.full_like(X_xz, yl[-1]), Z_xz, volume[:, -1, :])
    for iz in (0, len(zl) - 1):
        add_slice(X_xy, Y_xy, np.full_like(X_xy, zl[iz]), volume[iz])

    # Central transverse plane containing the selected particle position.
    iz0 = int(np.argmin(np.abs(zl)))
    add_slice(X_xy, Y_xy, np.full_like(X_xy, zl[iz0]), volume[iz0], 0.86)

    # Vertical oblique slice containing the beam axis and selected maximum.
    radius = np.hypot(particle_x, particle_y)
    if radius == 0.0:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = particle_x / radius, particle_y / radius
    s_limit = min(max(abs(xl).max(), abs(yl).max()), 1.0e-6)
    s = np.linspace(-s_limit, s_limit, 151)
    S, Z_oblique = np.meshgrid(s, zl, indexing="xy")
    X_oblique, Y_oblique = ux * S, uy * S
    interpolator = RegularGridInterpolator(
        (zl, yl, xl), volume, bounds_error=False, fill_value=np.nan
    )
    query = np.column_stack(
        (Z_oblique.ravel(), Y_oblique.ravel(), X_oblique.ravel())
    )
    I_oblique = interpolator(query).reshape(Z_oblique.shape)
    add_slice(X_oblique, Y_oblique, Z_oblique, I_oblique, 0.86)

    # The 0.99 isosurface marks the continuous helical intensity maxima.
    verts, faces, _, _ = marching_cubes(
        volume, level=isovalue,
        spacing=(zl[1] - zl[0], yl[1] - yl[0], xl[1] - xl[0]),
    )
    verts += np.array([zl[0], yl[0], xl[0]])
    xyz = verts[:, [2, 1, 0]]
    fig.add_trace(go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#b80000", opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.72, diffuse=0.65, specular=0.25,
                      roughness=0.5, fresnel=0.05),
        hoverinfo="skip", showscale=False,
    ))

    # Light-seeking dielectric particle placed at a center-plane maximum.
    theta, polar = np.mgrid[0:2*np.pi:101j, 0:np.pi:51j]
    sphere_radius = 1.5e-7
    sphere_x = particle_x + sphere_radius*np.cos(theta)*np.sin(polar)
    sphere_y = particle_y + sphere_radius*np.sin(theta)*np.sin(polar)
    sphere_z = particle_z + sphere_radius*np.cos(polar)
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        surfacecolor=np.ones_like(sphere_x),
        colorscale=[[0, "#008c20"], [1, "#00ff38"]],
        cmin=0, cmax=1, showscale=False,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.65,
                      roughness=0.25, fresnel=0.15),
        hovertemplate="Particle<extra></extra>",
    ))

    axis_style = dict(
        showbackground=False, showgrid=False, zeroline=False,
        exponentformat="power", showexponent="all",
    )
    fig.update_layout(
        width=900, height=1200, paper_bgcolor=background,
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        scene=dict(
            bgcolor=background,
            xaxis=dict(axis_style, title="x (m)", range=[xl.min(), xl.max()]),
            yaxis=dict(axis_style, title="y (m)", range=[yl.min(), yl.max()]),
            zaxis=dict(axis_style, title="z (m)", range=[zl.min(), zl.max()]),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.45, y=-1.45, z=1.05)),
        ),
    )

    script_dir = Path(__file__).resolve().parent
    html_path = script_dir / html_name
    fig.write_html(html_path, include_plotlyjs=True, auto_open=False)
    print(f"Interactive WebGL figure saved to: {html_path}")
    if png_name:
        png_path = script_dir / png_name
        try:
            fig.write_image(png_path, width=1200, height=1600, scale=2)
            print(f"High-resolution PNG saved to: {png_path}")
        except Exception as exc:
            print(f"PNG export skipped; use the complete HTML figure ({exc}).")
    fig.show()
    return fig


if __name__ == "__main__":
    simulation = run_15khz_simulation(show=True)
    rates = simulation["paper_pattern_rates"]
    print(f"VzTheory = {simulation['VzTheory']:.12g} m/s")
    print(f"WpTheory = {simulation['WpTheory']:.12g} rad/s")
    print(
        "Paper axial pattern speed = "
        f"{rates['axial_speed_m_per_s']:.12g} m/s"
    )
    print(
        "Paper rotation rate = "
        f"{rates['rotation_rate_rev_per_s']:.12g} rev/s"
    )
