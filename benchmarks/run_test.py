"""Binary k_ij fitting benchmark.

For each of 3 systems (mibk/water, 2m2boh/water, 2m3b2oh/water) × all water
models × 3 fitting scenarios (LLE-only, VLE-only, VLE+LLE) × order 0 and 1,
fits k_ij and reports ARD% at the fitted polynomial for BOTH data types,
including cross-evaluation (e.g. LLE-fitted k_ij evaluated against VLE data).

Each (water model, system) group runs the per-point k_ij fits once per data
type, then fits every scenario's k_ij(T) polynomial from them -- exactly what
``BinaryKijFitter.fit()`` does per combination, since its per-point stage does
not depend on order or scenario. Groups run in parallel worker processes.

Usage::

    uv run python benchmarks/run_test.py
"""

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fit_pcsaft._binary._utils import _fit_kij_polynomial
from fit_pcsaft._binary.fitter import _ard_at_coeffs
from fit_pcsaft._binary.lle import fit_kij_lle
from fit_pcsaft._binary.vle import fit_kij_vle

TEST_DIR = Path(__file__).parent
PARAMS_DIR = Path(__file__).parent.parent / "examples" / "data" / "parameters"

PURE_PARAMS = TEST_DIR / "test_pure.json"
WATER_MODELS_PATH = PARAMS_DIR / "water_models.json"
LOG_PATH = TEST_DIR / "run_test.log"

# --------------------------------------------------------------------------- #
# Water model names
# --------------------------------------------------------------------------- #
water_records = json.loads(WATER_MODELS_PATH.read_text(encoding="utf-8"))
WATER_MODELS = [r["identifier"]["name"] for r in water_records]

# --------------------------------------------------------------------------- #
# System definitions
# --------------------------------------------------------------------------- #
SYSTEMS = [
    {
        "name": "mibk_water",
        "id1": "mibk",
        "vle_path": TEST_DIR / "vle" / "mibk_water.csv",
        "lle_path": TEST_DIR / "lle" / "mibk_water.csv",
        "induced_assoc": True,
        "require_both_phases": False,
        "kij_t_ref": 298.15,
    },
    {
        "name": "2m2boh_water",
        "id1": "2-methyl-2-butanol",
        "vle_path": TEST_DIR / "vle" / "2m2boh_water.csv",
        "lle_path": TEST_DIR / "lle" / "2m2boh_water.csv",
        "induced_assoc": False,
        "require_both_phases": False,
        "kij_t_ref": 298.15,
    },
    {
        "name": "2m3b2oh_water",
        "id1": "2-methyl-3-buten-2-ol",
        "vle_path": TEST_DIR / "vle" / "2m3b2oh_water.csv",
        "lle_path": TEST_DIR / "lle" / "2m3b2oh_water.csv",
        "induced_assoc": False,
        "require_both_phases": False,
        "kij_t_ref": 298.15,
    },
]

SCENARIOS = ["lle", "vle", "vle+lle"]
ORDERS = [0, 1]
KIJ_BOUNDS = (-0.3, 0.3)  # BinaryKijFitter's default

PARAMS = [PURE_PARAMS, WATER_MODELS_PATH]


def _init_worker() -> None:
    import feos

    feos.set_num_threads(1)


def _run_group(water_name: str, sys: dict) -> "list[tuple[dict, str]]":
    """All scenario × order rows for one (water model, system), each with its log line."""
    id1, id2, t_ref = sys["id1"], water_name, sys["kij_t_ref"]
    common = {
        "kij_order": 0, "kij_t_ref": t_ref, "kij_bounds": KIJ_BOUNDS,
        "kij_per_point": True, "induced_assoc": sys["induced_assoc"],
    }
    # Per-point stage, once per data type; LLE first, matching the source
    # order BinaryKijFitter concatenated for vle+lle.
    srcs = {"lle": {"require_both_phases": sys["require_both_phases"]}, "vle": {}}
    sub: dict = {}
    for stype, fit_fn in (("lle", fit_kij_lle), ("vle", fit_kij_vle)):
        try:
            sub[stype] = fit_fn(id1, id2, sys[f"{stype}_path"], PARAMS, **common, **srcs[stype])
        except Exception as exc:  # noqa: BLE001 -- recorded on every row that needs it
            sub[stype] = exc

    out = []
    for scenario in SCENARIOS:
        types = [t for t in ("lle", "vle") if t in scenario]
        for order in ORDERS:
            tag = f"{sys['name']:20s}  {water_name:30s}  {scenario:8s}  ord={order}"
            row = {"system": sys["name"], "water": water_name, "scenario": scenario, "order": order}
            try:
                for t in types:
                    if isinstance(sub[t], Exception):
                        raise sub[t]
                fitted = [sub[t].data for t in types]
                kij_coeffs, _ = _fit_kij_polynomial(
                    np.concatenate([d["T_kij"] for d in fitted]),
                    np.concatenate([d["kij_pointwise"] for d in fitted]),
                    np.concatenate([d["ard_pointwise"] for d in fitted]),
                    order, t_ref,
                )
                # Both types at the polynomial: fitted ones as fit() reports
                # them, the unfitted one as cross-evaluation (NaN if its own
                # per-point fit failed, since that is where its data loads).
                ards = {
                    t: (float("nan") if isinstance(r, Exception)
                        else _ard_at_coeffs(r, kij_coeffs, t_ref, srcs[t]))
                    for t, r in sub.items()
                }
                counts = {t: len(sub[t].data["T_kij"]) for t in types}
                ard = sum(ards[t] * counts[t] for t in types) / sum(counts.values())
                row.update(
                    ard=ard, ard_lle=ards["lle"], ard_vle=ards["vle"],
                    kij0=float(kij_coeffs[0]),
                    kij1=float(kij_coeffs[1]) if len(kij_coeffs) > 1 else float("nan"),
                )
                msg = f"  OK        {tag}  LLE={ards['lle']:.2f}%  VLE={ards['vle']:.2f}%"
            except Exception as exc:  # noqa: BLE001 -- solver failure recorded as a NaN row
                row.update({k: float("nan") for k in ("ard", "ard_lle", "ard_vle", "kij0", "kij1")})
                msg = f"  FAIL      {tag}  {exc}"
            out.append((row, msg))
    return out


def main() -> None:
    groups = [(w, s) for w in WATER_MODELS for s in SYSTEMS]
    by_group: dict[int, list] = {}
    # spawn: forking a process with feos' live rayon threads risks a deadlock
    with (
        LOG_PATH.open("w", buffering=1, encoding="utf-8") as log,
        ProcessPoolExecutor(
            max_workers=max(1, (os.cpu_count() or 3) - 2),
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
        ) as pool,
        tqdm(total=len(groups) * len(SCENARIOS) * len(ORDERS),
             desc="Fitting", unit="fit", dynamic_ncols=True) as pbar,
    ):
        futures = {pool.submit(_run_group, w, s): i for i, (w, s) in enumerate(groups)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                by_group[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 -- e.g. a worker died
                w, s = groups[i]
                by_group[i] = [
                    ({"system": s["name"], "water": w, "scenario": scn, "order": o,
                      **dict.fromkeys(("ard", "ard_lle", "ard_vle", "kij0", "kij1"), float("nan"))},
                     f"  FAIL      {s['name']:20s}  {w:30s}  {scn:8s}  ord={o}  {exc}")
                    for scn in SCENARIOS for o in ORDERS
                ]
            for _, msg in by_group[i]:
                print(msg, file=log, flush=True)
                tqdm.write(msg)
            pbar.update(len(by_group[i]))

    results = [row for i in sorted(by_group) for row, _ in by_group[i]]
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
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


if __name__ == "__main__":
    main()
