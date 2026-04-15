"""Binary k_ij fitting benchmark.

For each of 3 systems (mibk/water, 2m2boh/water, 2m3b2oh/water) × all water
models × 3 fitting scenarios (LLE-only, VLE-only, VLE+LLE) × order 0 and 1,
fits k_ij and reports ARD% at the fitted polynomial for BOTH data types,
including cross-evaluation (e.g. LLE-fitted k_ij evaluated against VLE data).

Usage::

    uv run python examples/binary_test/run_test.py
"""

import json
from pathlib import Path

import numpy as np
import si_units as si
from tqdm import tqdm

from fit_pcsaft import BinaryKijFitter
from fit_pcsaft._binary._utils import (
    _apply_induced_association,
    _build_binary_eos,
    _load_pure_records,
)
from fit_pcsaft._binary.lle import (
    _LLE_FEEDS,
    _aggregate_lle_data,
    _exp_feeds,
    _residuals_at_T,
)
from fit_pcsaft._binary.vle import _residuals_vle_point
from fit_pcsaft._binary._utils import _kij_at_T
from fit_pcsaft._csv import SCHEMA_LLE, SCHEMA_VLE, load_csv

TEST_DIR = Path(__file__).parent
PARAMS_DIR = Path(__file__).parent.parent / "data" / "parameters"

PURE_PARAMS = TEST_DIR / "test_pure.json"
WATER_MODELS_PATH = PARAMS_DIR / "water_models.json"

# --------------------------------------------------------------------------- #
# Water model names
# --------------------------------------------------------------------------- #
water_records = json.loads(WATER_MODELS_PATH.read_text(encoding="utf-8"))
WATER_MODELS = [r["identifier"]["name"] for r in water_records]

# --------------------------------------------------------------------------- #
# System definitions
# --------------------------------------------------------------------------- #
SYSTEMS = [
    dict(
        name="mibk_water",
        id1="mibk",
        vle_path=TEST_DIR / "vle" / "mibk_water.csv",
        lle_path=TEST_DIR / "lle" / "mibk_water.csv",
        induced_assoc=True,
        require_both_phases=False,
        kij_t_ref=298.15,
    ),
    dict(
        name="2m2boh_water",
        id1="2-methyl-2-butanol",
        vle_path=TEST_DIR / "vle" / "2m2boh_water.csv",
        lle_path=TEST_DIR / "lle" / "2m2boh_water.csv",
        induced_assoc=False,
        require_both_phases=False,
        kij_t_ref=298.15,
    ),
    dict(
        name="2m3b2oh_water",
        id1="2-methyl-3-buten-2-ol",
        vle_path=TEST_DIR / "vle" / "2m3b2oh_water.csv",
        lle_path=TEST_DIR / "lle" / "2m3b2oh_water.csv",
        induced_assoc=False,
        require_both_phases=False,
        kij_t_ref=298.15,
    ),
]

SCENARIOS = ["lle", "vle", "vle+lle"]
ORDERS = [0, 1]

# --------------------------------------------------------------------------- #
# Cross-evaluation helpers
# Evaluate a pre-fitted k_ij polynomial against a data type WITHOUT re-fitting.
# --------------------------------------------------------------------------- #

def _eval_vle(
    id1, id2, params, kij_coeffs, kij_t_ref, vle_path, induced_assoc
) -> float:
    """ARD% at the given k_ij polynomial evaluated on VLE data (no re-fitting)."""
    record1, record2 = _load_pure_records(params, id1, id2)
    if induced_assoc:
        record1, record2 = _apply_induced_association(record1, record2)

    data = load_csv(vle_path, SCHEMA_VLE)
    T_arr = data["T"].astype(float)
    P_arr = data["P"].astype(float)
    x1_arr = data["x1"].astype(float)
    has_y1 = "y1" in data
    y1_arr = data["y1"].astype(float) if has_y1 else None

    # drop pure-component endpoints (same as fit_kij_vle)
    mix_mask = (x1_arr > 1e-4) & (x1_arr < 1.0 - 1e-4)
    T_arr = T_arr[mix_mask]
    P_arr = P_arr[mix_mask]
    x1_arr = x1_arr[mix_mask]
    if y1_arr is not None:
        y1_arr = y1_arr[mix_mask]

    ards = []
    for i in range(len(T_arr)):
        T_i = float(T_arr[i])
        kij_i = _kij_at_T(kij_coeffs, T_i, kij_t_ref)
        try:
            resid = _residuals_vle_point(
                [kij_i],
                T_i,
                float(P_arr[i]),
                float(x1_arr[i]),
                float(y1_arr[i]) if y1_arr is not None else None,
                record1,
                record2,
                si.KELVIN,
                si.KILO * si.PASCAL,
            )
            if resid is not None and len(resid) > 0:
                ards.append(100.0 * float(np.mean(np.abs(resid))))
        except Exception:
            pass

    return float(np.mean(ards)) if ards else float("nan")


def _eval_lle(
    id1, id2, params, kij_coeffs, kij_t_ref, lle_path, induced_assoc,
    require_both_phases=False
) -> float:
    """ARD% at the given k_ij polynomial evaluated on LLE data (no re-fitting)."""
    record1, record2 = _load_pure_records(params, id1, id2)
    if induced_assoc:
        record1, record2 = _apply_induced_association(record1, record2)

    raw = load_csv(lle_path, SCHEMA_LLE)
    T_arr = raw["T"].astype(float)
    x1_I = raw["x1_I"].astype(float) if "x1_I" in raw else None
    x1_II = raw["x1_II"].astype(float) if "x1_II" in raw else None

    aggregated = _aggregate_lle_data(T_arr, x1_I, x1_II, t_scale=1.0)
    if require_both_phases:
        aggregated = [
            (T, xi, xii) for T, xi, xii in aggregated
            if xi is not None and xii is not None
        ]

    if not aggregated:
        return float("nan")

    T_anchor_K = min(t for t, _, __ in aggregated)
    pressure = 1.01325 * si.BAR

    ards = []
    for T_K, exp_I, exp_II in aggregated:
        kij_i = _kij_at_T(kij_coeffs, T_K, kij_t_ref)
        feeds = _exp_feeds(exp_I, exp_II) + _LLE_FEEDS
        try:
            resid = _residuals_at_T(
                [kij_i], T_K, exp_I, exp_II,
                record1, record2, pressure, feeds,
                T_anchor_K=T_anchor_K,
            )
            if resid is not None and len(resid) > 0:
                ards.append(100.0 * float(np.mean(np.abs(resid))))
        except Exception:
            pass

    return float(np.mean(ards)) if ards else float("nan")


# --------------------------------------------------------------------------- #
# Run fits
# --------------------------------------------------------------------------- #
Row = dict
results: list[Row] = []

params = [PURE_PARAMS, WATER_MODELS_PATH]

combos = [
    (water_name, sys, scenario, order)
    for water_name in WATER_MODELS
    for sys in SYSTEMS
    for scenario in SCENARIOS
    for order in ORDERS
]

LOG_PATH = TEST_DIR / "run_test.log"
_log = LOG_PATH.open("w", buffering=1, encoding="utf-8")  # line-buffered

def _log_write(msg: str) -> None:
    print(msg, file=_log, flush=True)
    tqdm.write(msg)

with tqdm(combos, desc="Fitting", unit="fit", dynamic_ncols=True) as pbar:
    for water_name, sys, scenario, order in pbar:
        pbar.set_postfix_str(f"{sys['name']} | {water_name} | {scenario} | ord={order}")
        id1 = sys["id1"]
        id2 = water_name
        tag = f"{sys['name']:20s}  {water_name:30s}  {scenario:8s}  ord={order}"

        try:
            fitter = BinaryKijFitter(
                id1,
                id2,
                params,
                kij_order=order,
                kij_t_ref=sys["kij_t_ref"],
                induced_assoc=sys["induced_assoc"],
            )
            if "lle" in scenario:
                fitter.add_lle(
                    sys["lle_path"],
                    require_both_phases=sys["require_both_phases"],
                )
            if "vle" in scenario:
                fitter.add_vle(sys["vle_path"])

            _log_write(f"  fitting   {tag} ...")
            res = fitter.fit()
            _log_write(f"  fit done  {tag}  kij0={res.kij_coeffs[0]:.4f}  ARD={res.ard:.2f}%")

            kij_coeffs = res.kij_coeffs
            kij_t_ref = sys["kij_t_ref"]

            # ARD at polynomial for the fitted type(s) (post-poly, from fitter)
            ard_lle_fit = float(res.data["ard_lle"][0]) if "ard_lle" in res.data else float("nan")
            ard_vle_fit = float(res.data["ard_vle"][0]) if "ard_vle" in res.data else float("nan")

            # Cross-evaluation: evaluate fitted polynomial on the unfitted type
            if scenario == "lle":
                _log_write(f"  xeval VLE {tag} ...")
                ard_vle_fit = _eval_vle(
                    id1, id2, params, kij_coeffs, kij_t_ref,
                    sys["vle_path"], sys["induced_assoc"]
                )
                _log_write(f"  xeval done {tag}  ARD_VLE={ard_vle_fit:.2f}%")
            elif scenario == "vle":
                _log_write(f"  xeval LLE {tag} ...")
                ard_lle_fit = _eval_lle(
                    id1, id2, params, kij_coeffs, kij_t_ref,
                    sys["lle_path"], sys["induced_assoc"],
                    sys["require_both_phases"]
                )
                _log_write(f"  xeval done {tag}  ARD_LLE={ard_lle_fit:.2f}%")
            # for "vle+lle" both are already filled from the fitter

            results.append(dict(
                system=sys["name"],
                water=water_name,
                scenario=scenario,
                order=order,
                ard=res.ard,
                ard_lle=ard_lle_fit,
                ard_vle=ard_vle_fit,
                kij0=float(kij_coeffs[0]),
                kij1=float(kij_coeffs[1]) if len(kij_coeffs) > 1 else float("nan"),
            ))
            _log_write(f"  OK        {tag}  LLE={ard_lle_fit:.2f}%  VLE={ard_vle_fit:.2f}%")

        except Exception as exc:
            results.append(dict(
                system=sys["name"],
                water=water_name,
                scenario=scenario,
                order=order,
                ard=float("nan"),
                ard_lle=float("nan"),
                ard_vle=float("nan"),
                kij0=float("nan"),
                kij1=float("nan"),
            ))
            _log_write(f"  FAIL      {tag}  {exc}")

_log.close()

# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
W_SYS = 22
W_WAT = 32
W_SCN = 10
W_ORD = 6
W_VAL = 12
SEP = "-" * (W_SYS + W_WAT + W_SCN + W_ORD + W_VAL * 3 + 4)

print(f"\n{'='*len(SEP)}")
print("SUMMARY  (ARD at polynomial in %;  cross-eval shown for single-type scenarios)")
print(f"{'='*len(SEP)}")
print(
    f"{'System':<{W_SYS}} {'Water model':<{W_WAT}} {'Scenario':<{W_SCN}} "
    f"{'Order':<{W_ORD}} {'ARD_LLE%':>{W_VAL}} {'ARD_VLE%':>{W_VAL}} {'kij0':>{W_VAL}}"
)
print(SEP)

prev_sys = None
prev_water = None
for r in results:
    if r["system"] != prev_sys or r["water"] != prev_water:
        if prev_sys is not None:
            print()
        prev_sys = r["system"]
        prev_water = r["water"]

    lle_str = f"{r['ard_lle']:.2f}" if not np.isnan(r["ard_lle"]) else "FAIL"
    vle_str = f"{r['ard_vle']:.2f}" if not np.isnan(r["ard_vle"]) else "FAIL"
    kij0_str = f"{r['kij0']:.4f}" if not np.isnan(r["kij0"]) else "FAIL"

    print(
        f"{r['system']:<{W_SYS}} {r['water']:<{W_WAT}} {r['scenario']:<{W_SCN}} "
        f"{r['order']:<{W_ORD}} {lle_str:>{W_VAL}} {vle_str:>{W_VAL}} {kij0_str:>{W_VAL}}"
    )
