"""Investigate the UCST of the 1-hexanol+water system as a function of k_ij.

Key question: Does the model predict LLE extending to 493K for any k_ij?
If the UCST is below 433K for k_ij=0.0, we need a large negative k_ij to extend it.
"""
import numpy as np
import feos
import si_units as si
import json, tempfile, os

water_record_dict = {
    "identifier": {"cas": "7732-18-5", "name": "water"},
    "molarweight": 18.015,
    "m": 1.0656,
    "sigma": 3.0007,
    "epsilon_k": 366.51,
    "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.034868, "epsilon_k_ab": 2500.7}]
}
hexanol_record_dict = {
    "identifier": {"cas": "111-27-3", "name": "1-hexanol"},
    "molarweight": 102.175,
    "m": 3.3312,
    "sigma": 3.7483,
    "epsilon_k": 270.86,
    "association_sites": [{"na": 1.0, "nb": 1.0, "kappa_ab": 0.002566, "epsilon_k_ab": 2778.9}]
}

def build_eos(kij=0.0):
    combined = [water_record_dict, hexanol_record_dict]
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
    json.dump(combined, tmp)
    tmp.close()
    params = feos.Parameters.from_json(["water", "1-hexanol"], pure_path=tmp.name)
    os.unlink(tmp.name)
    binary_params = feos.Parameters.new_binary(params.pure_records, k_ij=kij)
    return feos.EquationOfState.pcsaft(binary_params, max_iter_cross_assoc=100)

pressure = 1.0 * si.BAR

print("=" * 70)
print("TEST 1: Find UCST as function of k_ij using PhaseDiagram.lle")
print("=" * 70)

kij_values = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
feeds_to_try = [0.10, 0.15, 0.20, 0.25, 0.30, 0.05, 0.40, 0.50]

for kij in kij_values:
    eos = build_eos(kij=kij)
    ucst = None
    pd_obj = None
    for feed_z1 in feeds_to_try:
        for t_max_K in [520.0, 500.0, 480.0]:
            try:
                feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
                pd = feos.PhaseDiagram.lle(
                    eos, pressure, feed,
                    min_tp=270.0 * si.KELVIN,
                    max_tp=t_max_K * si.KELVIN,
                    npoints=200
                )
                T_arr = pd.liquid.temperature / si.KELVIN
                if len(T_arr) > 0:
                    ucst = T_arr.max()
                    pd_obj = pd
                    break
            except Exception as e:
                pass
        if ucst is not None:
            break
    if ucst is not None:
        T_arr = pd_obj.liquid.temperature / si.KELVIN
        x1_I = pd_obj.liquid.molefracs[:, 0]
        x1_II = pd_obj.vapor.molefracs[:, 0]
        print(f"k_ij={kij:+.3f}: UCST={ucst:.1f}K, {len(T_arr)} pts, x1_I range=[{x1_I.min():.4f},{x1_I.max():.4f}], x1_II=[{x1_II.min():.4f},{x1_II.max():.4f}]")
    else:
        print(f"k_ij={kij:+.3f}: PhaseDiagram.lle FAILED for all feeds")

print()
print("=" * 70)
print("TEST 2: What does tp_flash see at T=430-493K for k_ij=-0.1?")
print("=" * 70)

eos_neg = build_eos(kij=-0.1)
for T_K in [430.0, 443.2, 453.2, 463.2, 473.2, 483.2, 493.2]:
    feeds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    found = False
    for z1 in feeds:
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pe = feos.PhaseEquilibrium.tp_flash(eos_neg, T_K * si.KELVIN, pressure, feed)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) > 1e-4:
                print(f"T={T_K:.1f}K, k_ij=-0.1: SUCCESS feed={z1:.3f} → x1_I={min(x_a,x_b):.5f}, x1_II={max(x_a,x_b):.5f}")
                found = True
                break
        except:
            pass
    if not found:
        # check stability
        try:
            feed = np.array([0.10, 0.90]) * si.MOL
            state = feos.State(eos_neg, T_K * si.KELVIN, pressure=pressure, moles=feed)
            print(f"T={T_K:.1f}K, k_ij=-0.1: FAILED, stability z1=0.10: {state.is_stable()}")
        except:
            print(f"T={T_K:.1f}K, k_ij=-0.1: FAILED")

print()
print("=" * 70)
print("TEST 3: Critical point of the binary mixture (finds UCST precisely)")
print("=" * 70)
print("Using feos.State.critical_point_binary")

for kij in [0.0, -0.1, -0.2]:
    eos = build_eos(kij=kij)
    # Try compositions near the critical point (symmetric-ish)
    for z1 in [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        try:
            moles = np.array([z1, 1.0 - z1]) * si.MOL
            cp = feos.State.critical_point_binary(eos, moles=moles)
            T_cp = float(cp.temperature / si.KELVIN)
            P_cp = float(cp.pressure / si.BAR)
            x1_cp = float(cp.molefracs[0])
            print(f"k_ij={kij:+.3f}, z1={z1:.2f}: UCST={T_cp:.1f}K, P={P_cp:.3f}bar, x1={x1_cp:.4f}")
            break
        except Exception as e:
            pass
    else:
        print(f"k_ij={kij:+.3f}: critical_point_binary failed for all compositions")
