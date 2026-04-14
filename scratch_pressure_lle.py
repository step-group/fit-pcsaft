"""Toy script: effect of pressure on LLE phase compositions (water + toluene).

Fixes T and k_ij; sweeps pressure from 1 bar to 1000 bar;
prints x1_I and x1_II for each pressure.
"""

from pathlib import Path

import feos
import numpy as np
import si_units as si

from fit_pcsaft._binary._utils import _build_binary_eos, _load_pure_records

# --- load records ---------------------------------------------------------
params_path = Path("examples/data/parameters/binary_params.json")

record_toluene, record_water = _load_pure_records(params_path, "toluene", "water")

KIJ = 0.0
print(f"Using k_ij = {KIJ:.4f}")

eos = _build_binary_eos(record_toluene, record_water, KIJ)

# --------------------------------------------------------------------------
T_K = 298.15  # fix temperature
feed = np.array([0.5, 0.5]) * si.MOL  # 50/50 feed

pressures_bar = [1, 10, 50, 100, 200, 500, 1000]


def log_odds(x):
    return np.log10(x / (1.0 - x))


print(f"\nT = {T_K} K,  feed = 50/50 (toluene/water)")
print(
    f"{'P [bar]':>10}  {'x1_I':>12}  {'log(x1_I)':>12}  {'x1_II':>12}  {'log(x1_II)':>12}"
)
print("-" * 68)

for p_bar in pressures_bar:
    p = p_bar * si.BAR
    try:
        state = feos.State(
            eos,
            T_K * si.KELVIN,
            pressure=p,
            moles=feed,
            density_initialization="liquid",
        )
        pe = state.tp_flash(max_iter=10000)
        x_a = float(pe.liquid.molefracs[0])
        x_b = float(pe.vapor.molefracs[0])
        if abs(x_a - x_b) < 1e-6:
            print(f"{p_bar:>10}  {'(single phase)':>12}")
            continue
        x_I, x_II = min(x_a, x_b), max(x_a, x_b)
        print(
            f"{p_bar:>10}  {x_I:>12.6f}  {log_odds(x_I):>12.4f}  {x_II:>12.6f}  {log_odds(x_II):>12.4f}"
        )
    except Exception as e:
        print(f"{p_bar:>10}  ERROR: {e}")
