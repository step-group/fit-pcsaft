"""Investigate the quality of tp_flash predictions vs experimental data.

At low T (273-363K), tp_flash with k_ij=0 predicts x1_I ~0.32-0.55 (!)
while experiments show x1_I ~0.001 (water-lean) and x1_II ~0.69-0.72 (hexanol-rich).
This means tp_flash is finding the wrong phase split at low T.

Also: Does PhaseDiagram.lle cover the full low-T range better?
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
_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
SIGMOID_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()
EXTRA_FEEDS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
ALL_FEEDS = SIGMOID_FEEDS + EXTRA_FEEDS

print("=" * 70)
print("TEST 1: tp_flash predictions vs experiment at k_ij=0 (low T region)")
print("=" * 70)
print(f"{'T(K)':>6} {'pred_x1_I':>10} {'pred_x1_II':>11} {'exp_x1_I':>9} {'exp_x1_II':>10}")

eos0 = build_eos(0.0)
exp_low_T = [
    (273.2, 0.00139, 0.7172),
    (283.2, 0.0012, 0.7205),
    (293.2, 0.0011, 0.6944),
    (303.2, 0.001, 0.701),
    (323.2, 0.0009, 0.6898),
    (333.2, 0.0009, 0.6688),
    (343.2, 0.001, 0.6464),
    (353.2, 0.0011, 0.6228),
    (363.2, 0.0012, 0.6018),
    (373.2, 0.0014, 0.5806),
    (383.2, 0.0016, 0.5582),
    (393.2, 0.0019, 0.5348),
    (403.2, 0.0021, 0.5098),
]

for T_K, exp_I, exp_II in exp_low_T:
    found = False
    for z1 in ALL_FEEDS:
        z1 = float(np.clip(z1, 1e-4, 1-1e-4))
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pe = feos.PhaseEquilibrium.tp_flash(eos0, T_K * si.KELVIN, pressure, feed)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) > 1e-4:
                print(f"{T_K:6.1f} {min(x_a,x_b):10.6f} {max(x_a,x_b):11.6f} {exp_I:9.4f} {exp_II:10.4f}")
                found = True
                break
        except:
            pass
    if not found:
        print(f"{T_K:6.1f} {'FAILED':>10}")

print()
print("Note: At low T, tp_flash predicts x1_I ~0.3-0.5 (hexanol-rich water phase?)")
print("The water_gross2002 model likely gives wrong phase boundaries.")
print()

print("=" * 70)
print("TEST 2: Try k_ij=0.1, 0.2 — do these give better compositions?")
print("=" * 70)

for kij in [0.1, 0.2]:
    print(f"\nk_ij={kij}")
    eos_k = build_eos(kij)
    print(f"{'T(K)':>6} {'pred_x1_I':>10} {'pred_x1_II':>11} {'exp_x1_I':>9} {'exp_x1_II':>10}")
    for T_K, exp_I, exp_II in exp_low_T[:6]:
        for z1 in ALL_FEEDS:
            z1 = float(np.clip(z1, 1e-4, 1-1e-4))
            try:
                feed = np.array([z1, 1.0 - z1]) * si.MOL
                pe = feos.PhaseEquilibrium.tp_flash(eos_k, T_K * si.KELVIN, pressure, feed)
                x_a = float(pe.liquid.molefracs[0])
                x_b = float(pe.vapor.molefracs[0])
                if abs(x_a - x_b) > 1e-4:
                    print(f"{T_K:6.1f} {min(x_a,x_b):10.6f} {max(x_a,x_b):11.6f} {exp_I:9.4f} {exp_II:10.4f}")
                    break
            except:
                pass

print()
print("=" * 70)
print("TEST 3: PhaseDiagram.lle — does it cover the full low-T range?")
print("=" * 70)
print("(The PD.lle continuation method should follow the entire LLE curve)")
print()

for kij in [0.0, 0.1, 0.2]:
    eos_k = build_eos(kij)
    for feed_z1 in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        try:
            feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
            pd = feos.PhaseDiagram.lle(
                eos_k, pressure, feed,
                min_tp=270.0 * si.KELVIN,
                max_tp=500.0 * si.KELVIN,
                npoints=500
            )
            T_arr = pd.liquid.temperature / si.KELVIN
            x1_I = pd.liquid.molefracs[:, 0]
            x1_II = pd.vapor.molefracs[:, 0]
            if T_arr.min() < 280.0:
                print(f"k_ij={kij:+.2f}, feed={feed_z1:.2f}: {len(T_arr)} pts, T=[{T_arr.min():.1f},{T_arr.max():.1f}]K, x1_I=[{x1_I.min():.5f},{x1_I.max():.5f}]")
                # Show a sample of points
                step = max(1, len(T_arr) // 10)
                for i in range(0, len(T_arr), step):
                    print(f"    T={T_arr[i]:.1f}K, x1_I={x1_I[i]:.6f}, x1_II={x1_II[i]:.6f}")
                break
            else:
                print(f"k_ij={kij:+.2f}, feed={feed_z1:.2f}: {len(T_arr)} pts, T=[{T_arr.min():.1f},{T_arr.max():.1f}]K (doesn't reach 273K)")
        except Exception as e:
            print(f"k_ij={kij:+.2f}, feed={feed_z1:.2f}: FAILED ({e})")
    print()

print()
print("=" * 70)
print("TEST 4: Spinodal analysis at low T — confirm two-phase nature")
print("=" * 70)
print()
eos0 = build_eos(0.0)
print("Spinodal at T=298.2K:")
try:
    spinodal = feos.PhaseDiagram.spinodal(eos0, 298.2 * si.KELVIN)
    T_s = spinodal.states.temperature / si.KELVIN
    x1_s = spinodal.states.molefracs[:, 0]
    print(f"  Spinodal points: {len(x1_s)}")
    for i, (T, x) in enumerate(zip(T_s, x1_s)):
        print(f"    x1={x:.6f}, T={T:.1f}K")
except Exception as e:
    print(f"  Spinodal failed: {e}")
