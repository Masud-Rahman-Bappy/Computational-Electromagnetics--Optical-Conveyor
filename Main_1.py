"""Structured-Bessel-beam tractor-field simulation and 3-D visualization.

This program implements the optical landscape described by Rahman et al.,
"Tractor beam for fully immersed multiple objects: Long distance pulling,
trapping, and rotation with a single optical set-up," Annalen der Physik 527,
777-793 (2015), DOI 10.1002/andp.201500266.

Two higher-order Bessel beams with reverse orders produce ``2*abs(m)``
azimuthal trapping regions. Unequal longitudinal wave numbers produce axial
modulation, and a small frequency difference moves the interference landscape
without an externally ramped phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from skimage.measure import marching_cubes

from BesselBeamGasLiquidForceFunc import bessel_beam_gas_liquid_force
from OpticalForceAnalysisFunc import pattern_velocities


def find_nearest_intensity_maximum(
    volume, xl, yl, zl, z_target, preferred_xy=(-6e-7, 0.0),
):
    """Return an exact intensity maximum on the plane nearest ``z_target``.

    The reverse-order beam pair produces several symmetry-related maxima.  The
    candidate nearest ``preferred_xy`` is selected so the same helical branch
    is tracked while the interference landscape moves.
    """
    iz = int(np.argmin(np.abs(zl - z_target)))
    plane = np.asarray(volume[iz], dtype=float)
    plane_max = float(np.nanmax(plane))
    local_maximum = plane == maximum_filter(plane, size=5, mode="nearest")
    candidates = np.argwhere(local_maximum & (plane >= 0.90 * plane_max))
    if candidates.size == 0:
        candidates = np.array([
            np.unravel_index(np.nanargmax(plane), plane.shape)
        ])

    preferred_x, preferred_y = preferred_xy
    candidate_x = xl[candidates[:, 1]]
    candidate_y = yl[candidates[:, 0]]
    distance_squared = (
        (candidate_x - preferred_x) ** 2 + (candidate_y - preferred_y) ** 2
    )
    selected = candidates[int(np.argmin(distance_squared))]
    iy, ix = int(selected[0]), int(selected[1])
    return {
        "x": float(xl[ix]), "y": float(yl[iy]), "z": float(zl[iz]),
        "ix": ix, "iy": iy, "iz": iz,
        "normalized_intensity": float(plane[iy, ix]),
    }


def run_simulation(nn=1, mm=1, tt=45.0, show=True):
    Ce, Ch = 2.5e6, 0.0
    wave_factor = 1.0
    delf, lamda = 5000.0, wave_factor * 1e-6
    Delw = delf / (3e8 / lamda)
    Delb, Parx = 0.45, 40.0
    order = np.array([2, -2])
    rngg = np.array([5.403245142485066e-7, 4.802389364443375e-7])
    direction = 1
    incident_angle = np.array([45.0, 45.0])

    # Particle and liquid parameters used by the paper's dipole-force model.
    # For the defaults, 2*pi*r_p/lambda < 1, satisfying the stated dipole limit.
    particle = dict(radius=wave_factor * 100e-9, density=2650.0,
                    permittivity=3.90 * 8.854187817e-12)
    medium = dict(density=1000.0, permittivity=1.7689 * 8.854187817e-12,
                  viscosity=1.002e-3)
    meu_g_r, epshilon_g_r = 1.0, 1.0
    meu_l_r, epshilon_l_r = 1.0, 1.7689
    t = tt * 3e-6

    edge = 1.5e-6 * wave_factor
    step = (1.5e-8 * wave_factor) / 2 ** (nn - 1)
    xy = np.arange(-edge, edge + 0.5 * step, step)
    z_step = (6e-8 * wave_factor) / 2 ** (mm - 1)
    zl = np.arange(-3e-6 * wave_factor, 3e-6 * wave_factor + 0.5 * z_step, z_step)

    int_l = np.empty((zl.size, xy.size, xy.size), dtype=np.float64)
    int_l_raw = np.empty_like(int_l)
    beta_values = np.empty(2)
    omega_values = np.empty(2)
    xl = yl = None

    for iz, z_value in enumerate(zl):
        totals = {key: 0.0 for key in ("gE_rho", "gE_phi", "gE_z", "lE_rho", "lE_tee", "lE_zee")}
        for loop in (1, 2):
            result = bessel_beam_gas_liquid_force(
                incident_angle, order[loop - 1], Ce, Ch, xy, xy, loop,
                z_value, t, Parx, Delw, Delb, meu_g_r, epshilon_g_r,
                meu_l_r, epshilon_l_r, lamda, rngg, direction,
            )
            xl, yl = result["xl"], result["yl"]
            for key in totals:
                totals[key] = totals[key] + result[key]
            beta_values[loop - 1], omega_values[loop - 1] = result["beta"], result["w"]
            if loop == 1:
                vz_theory, wp_theory = result["VzTheory"], result["WpTheory"]

        # Coherent superposition of the two liquid fields (paper Eq. 5a),
        # followed by |E_rho|^2 + |E_phi|^2 + |E_z|^2 (paper Eq. 5b).
        raw = sum(np.abs(totals[k]) ** 2 for k in ("lE_rho", "lE_tee", "lE_zee"))
        int_l_raw[iz] = raw
        int_l[iz] = raw / raw.max()

    angle = np.arctan2(vz_theory, wp_theory)
    vz_particle = vz_theory * np.cos(angle) ** 2
    wp_particle = wp_theory * np.sin(angle) ** 2
    paper_rates = pattern_velocities(omega_values, beta_values, order)

    # Place the light-seeking particle at an actual intensity maximum on the
    # same longitudinal plane used for the particle visualization.
    particle_z_target = float(np.clip(abs(vz_particle) * t, zl.min(), zl.max()))
    particle_position = find_nearest_intensity_maximum(
        int_l, xl, yl, zl, particle_z_target,
    )

    if show:
        plot_intensity_volume(
            int_l, xl, yl, zl, vz_particle, t,
            particle_position=particle_position,
        )

    return {
        "IntL": int_l, "IntLsave": int_l_raw, "xl": xl, "yl": yl, "zl": zl,
        "VzTheory": vz_theory, "WpTheory": wp_theory,
        "VzParticle": vz_particle, "WpParticle": wp_particle,
        "BETAvalue": beta_values, "OMEGAvalue": omega_values,
        "paper_pattern_rates": paper_rates,
        "particle_position": particle_position,
        "particle": particle, "medium": medium,
    }


def plot_intensity_volume(
    volume, xl, yl, zl, vz_particle, t, isovalue=0.98,
    save_path="HelicalPatternLiquid_Intensity_interactive.html",
    png_path="HelicalPatternLiquid_Intensity_high_quality.png",
    particle_position=None,
):
    """Render paper-style intensity slices, helical maxima, and particle."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly is required for the corrected 3-D renderer. Install it with: "
            "python -m pip install plotly kaleido"
        ) from exc

    background = "#b8e9f7"
    colorscale = [
        [0.00, "#242080"], [0.30, "#4154b8"],
        [0.52, "#7ed9ee"], [0.68, "#efffff"],
        [0.80, "#fff129"], [0.92, "#ff2517"],
        [1.00, "#8a0000"],
    ]
    fig = go.Figure()

    def add_slice(x, y, z, intensity, opacity=0.78):
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=np.asarray(intensity),
            colorscale=colorscale, cmin=0.0, cmax=1.0,
            opacity=opacity, showscale=False,
            hovertemplate="I/Imax=%{surfacecolor:.3f}<extra></extra>",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                          roughness=1.0, fresnel=0.0),
            contours=dict(x=dict(show=False), y=dict(show=False),
                          z=dict(show=False)),
        ))

    # Orthogonal slices expose the radial, azimuthal, and longitudinal
    # modulation discussed with Figs. 1(b), 2(c), and 2(d) of the paper.
    X_xy, Y_xy = np.meshgrid(xl, yl, indexing="xy")
    Y_yz, Z_yz = np.meshgrid(yl, zl, indexing="xy")
    X_xz, Z_xz = np.meshgrid(xl, zl, indexing="xy")

    # Two vertical x-normal slices: boundary and beam-axis planes.
    for ix in (0, int(np.argmin(np.abs(xl)))):
        add_slice(np.full_like(Y_yz, xl[ix]), Y_yz, Z_yz,
                  volume[:, :, ix])

    # One vertical y-normal boundary slice.
    iy = len(yl) - 1
    add_slice(X_xz, np.full_like(X_xz, yl[iy]), Z_xz,
              volume[:, iy, :])

    # Bottom slice and the slice through the displaced particle position.
    if particle_position is None:
        particle_z_target = float(
            np.clip(abs(vz_particle) * t, zl.min(), zl.max())
        )
        particle_position = find_nearest_intensity_maximum(
            volume, xl, yl, zl, particle_z_target,
        )
    particle_z = particle_position["z"]
    for iz in sorted({0, particle_position["iz"]}):
        add_slice(X_xy, Y_xy, np.full_like(X_xy, zl[iz]),
                  volume[iz, :, :])

    # Isosurface axes are ordered (z, y, x); convert coordinates to metres.
    verts, faces, _, _ = marching_cubes(
        volume, level=isovalue,
        spacing=(zl[1] - zl[0], yl[1] - yl[0], xl[1] - xl[0]),
    )
    verts += np.array([zl[0], yl[0], xl[0]])
    xyz_vertices = verts[:, [2, 1, 0]]  # marching cubes (z,y,x) -> plot (x,y,z)
    fig.add_trace(go.Mesh3d(
        x=xyz_vertices[:, 0], y=xyz_vertices[:, 1], z=xyz_vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="#b80000", opacity=1.0, flatshading=False,
        lighting=dict(ambient=0.72, diffuse=0.65, specular=0.25,
                      roughness=0.5, fresnel=0.05),
        lightposition=dict(x=2e-6, y=2e-6, z=5e-6),
        hoverinfo="skip", name=f"I/Imax = {isovalue:.2f}", showscale=False,
    ))

    # Particle sphere.
    u, v = np.mgrid[0:2*np.pi:101j, 0:np.pi:51j]
    radius = 1.8e-7
    xc, yc, zc = (
        particle_position["x"],
        particle_position["y"],
        particle_position["z"],
    )
    sphere_x = xc + radius*np.cos(u)*np.sin(v)
    sphere_y = yc + radius*np.sin(u)*np.sin(v)
    sphere_z = zc + radius*np.cos(v)
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        surfacecolor=np.ones_like(sphere_x),
        colorscale=[[0, "#00a000"], [1, "#00ff38"]],
        cmin=0, cmax=1, showscale=False, opacity=1.0,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.65,
                      roughness=0.25, fresnel=0.15),
        hovertemplate="Particle<extra></extra>", name="Particle",
    ))

    axis_style = dict(
        title_font=dict(size=18), tickfont=dict(size=14),
        showbackground=False, showgrid=False, zeroline=False,
        exponentformat="power", showexponent="all",
    )
    fig.update_layout(
        width=900, height=1200, paper_bgcolor=background,
        plot_bgcolor=background, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        scene=dict(
            bgcolor=background,
            xaxis=dict(axis_style, title="x (m)", range=[xl.min(), xl.max()]),
            yaxis=dict(axis_style, title="y (m)", range=[yl.min(), yl.max()]),
            zaxis=dict(axis_style, title="z (m)", range=[zl.min(), zl.max()]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.05),
                        up=dict(x=0, y=0, z=1)),
        ),
    )

    if save_path:
        save_target = Path(save_path).expanduser()
        if not save_target.is_absolute():
            save_target = Path(__file__).resolve().parent / save_target
        fig.write_html(save_target, include_plotlyjs=True, auto_open=False)
        print(f"Interactive WebGL figure saved to: {save_target}")

    if png_path:
        png_target = Path(png_path).expanduser()
        if not png_target.is_absolute():
            png_target = Path(__file__).resolve().parent / png_target
        try:
            fig.write_image(png_target, width=1200, height=1600, scale=2)
            print(f"High-resolution PNG saved to: {png_target}")
        except Exception as exc:
            print("PNG export skipped. The interactive HTML is complete. "
                  f"Install/configure kaleido for PNG export ({exc}).")

    fig.show()
    return fig


if __name__ == "__main__":
    simulation = run_simulation()
    print(f"VzTheory = {simulation['VzTheory']:.12g}")
    print(f"WpTheory = {simulation['WpTheory']:.12g}")
    print(f"VzParticle = {simulation['VzParticle']:.12g}")
    print(f"WpParticle = {simulation['WpParticle']:.12g}")
    p = simulation["particle_position"]
    print(
        "Particle maximum: "
        f"x={p['x']:.12g} m, y={p['y']:.12g} m, z={p['z']:.12g} m, "
        f"I/Imax={p['normalized_intensity']:.12g}"
    )
    rates = simulation["paper_pattern_rates"]
    print(f"Paper axial pattern speed = {rates['axial_speed_m_per_s']:.12g} m/s")
    print(f"Paper rotation rate = {rates['rotation_rate_rev_per_s']:.12g} rev/s")
