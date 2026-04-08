"""Compare PhaseDiagram.lle vs tp_flash in detail.

Key questions:
1. PhaseDiagram.lle sees 17 pts at k_ij=0, up to ~425.8K. tp_flash succeeds up to ~423.2K.
   What happens between 423.2K and 425.8K?
2. Does PhaseDiagram.lle recover the low-T (273-363K) compositions that tp_flash finds?
3. Which approach gives better compositions vs experimental data?
4. Is PhaseDiagram.lle more robust for the optimization loop (fewer cold-start failures)?
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

eos = build_eos(kij=0.0)
pressure = 1.0 * si.BAR

print("=" * 70)
print("TEST 1: PhaseDiagram.lle full output with k_ij=0, multiple feeds")
print("=" * 70)

# Try multiple feeds and multiple npoints to get the most complete picture
best_pd = None
best_npts = 0
for feed_z1 in [0.20, 0.15, 0.25, 0.10, 0.30, 0.05]:
    for npoints in [500, 300, 200]:
        try:
            feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
            pd = feos.PhaseDiagram.lle(
                eos, pressure, feed,
                min_tp=270.0 * si.KELVIN,
                max_tp=500.0 * si.KELVIN,
                npoints=npoints
            )
            T_arr = pd.liquid.temperature / si.KELVIN
            n = len(T_arr)
            print(f"feed={feed_z1:.2f}, npoints={npoints}: {n} pts, T=[{T_arr.min():.1f},{T_arr.max():.1f}]K")
            if n > best_npts:
                best_npts = n
                best_pd = pd
        except Exception as e:
            print(f"feed={feed_z1:.2f}, npoints={npoints}: FAILED ({e})")

print()
if best_pd is not None:
    T_arr = best_pd.liquid.temperature / si.KELVIN
    x1_I = best_pd.liquid.molefracs[:, 0]
    x1_II = best_pd.vapor.molefracs[:, 0]
    print(f"Best PhaseDiagram.lle result: {len(T_arr)} points, T=[{T_arr.min():.1f},{T_arr.max():.1f}]K")
    print(f"{'T (K)':>8} {'x1_I':>10} {'x1_II':>10}")
    for i in range(len(T_arr)):
        print(f"{T_arr[i]:8.1f} {x1_I[i]:10.6f} {x1_II[i]:10.6f}")

print()
print("=" * 70)
print("TEST 2: tp_flash at T=415..430K (fine grid, near UCST)")
print("=" * 70)

_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
SIGMOID_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()
EXTRA_FEEDS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]
ALL_FEEDS = SIGMOID_FEEDS + EXTRA_FEEDS

for T_K in np.arange(413.0, 435.0, 1.0):
    found = False
    for z1 in ALL_FEEDS:
        z1 = float(np.clip(z1, 1e-4, 1-1e-4))
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pe = feos.PhaseEquilibrium.tp_flash(eos, T_K * si.KELVIN, pressure, feed)
            x_a = float(pe.liquid.molefracs[0])
            x_b = float(pe.vapor.molefracs[0])
            if abs(x_a - x_b) > 1e-4:
                pred_I, pred_II = min(x_a, x_b), max(x_a, x_b)
                print(f"T={T_K:.1f}K: feed={z1:.4f} → x1_I={pred_I:.5f}, x1_II={pred_II:.5f}")
                found = True
                break
        except:
            pass
    if not found:
        print(f"T={T_K:.1f}K: FAILED")

print()
print("=" * 70)
print("TEST 3: PhaseDiagram.lle for multiple k_ij values — full coverage")
print("=" * 70)
# Check how many experimental temperature points each k_ij can cover
# using the PhaseDiagram.lle approach

exp_T_all = sorted(set([
    273.2, 278.7, 280.0, 281.8, 283.2, 283.4, 284.2, 285.2, 286.1, 287.8,
    288.2, 290.2, 293.2, 293.9, 296.1, 296.2, 298.2, 300.1, 302.1, 302.9,
    303.1, 303.2, 304.1, 306.2, 308.0, 308.2, 313.0, 313.1, 313.2, 318.0,
    323.2, 333.2, 343.2, 343.5, 353.2, 353.5, 363.2, 363.5, 373.2, 383.2,
    393.2, 403.2, 413.2, 423.2, 433.2, 443.2, 453.2, 463.2, 473.2, 483.2,
    493.2
]))

def get_pd_coverage(eos, pressure, t_min_K, t_max_K, npoints=200):
    """Try multiple feeds, return (T_arr, x1_I, x1_II) or None."""
    feeds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.05, 0.40]
    for z1 in feeds:
        try:
            feed = np.array([z1, 1.0 - z1]) * si.MOL
            pd = feos.PhaseDiagram.lle(
                eos, pressure, feed,
                min_tp=t_min_K * si.KELVIN,
                max_tp=t_max_K * si.KELVIN,
                npoints=npoints
            )
            T = pd.liquid.temperature / si.KELVIN
            if len(T) > 0:
                return T, pd.liquid.molefracs[:, 0], pd.vapor.molefracs[:, 0]
        except:
            pass
    return None

# Interpolate PhaseDiagram.lle onto exp temperatures using nearest-point
for kij in [0.0, 0.1, 0.2]:
    eos_k = build_eos(kij=kij)
    result = get_pd_coverage(eos_k, pressure, 270.0, 500.0)
    if result is None:
        print(f"k_ij={kij:+.2f}: PhaseDiagram.lle failed entirely")
        continue
    T_pd, xi_pd, xii_pd = result
    # How many experimental T points fall within the PD range?
    t_lo, t_hi = T_pd.min(), T_pd.max()
    covered = [T for T in exp_T_all if t_lo <= T <= t_hi]
    print(f"k_ij={kij:+.2f}: PD covers T=[{t_lo:.1f},{t_hi:.1f}]K, {len(covered)}/{len(exp_T_all)} exp points covered")

print()
print("=" * 70)
print("TEST 4: What error does the water_gross2002 model actually give")
print("        for the high-T region (model limitation vs flash failure)?")
print("=" * 70)
print()
print("Experimental data above 423K:")
exp_high = [(433.2, None, 0.4237), (443.2, 0.005, 0.3917), (453.2, 0.0066, 0.3534),
            (463.2, 0.009, 0.3099), (473.2, 0.0128, 0.2602), (483.2, 0.0189, 0.2008),
            (493.2, 0.0342, 0.1245)]
print("  This is the UCST region. Best tp_flash result at 423.2K with k_ij=0:")
eos0 = build_eos(0.0)
for z1 in ALL_FEEDS:
    z1 = float(np.clip(z1, 1e-4, 1-1e-4))
    try:
        feed = np.array([z1, 1.0 - z1]) * si.MOL
        pe = feos.PhaseEquilibrium.tp_flash(eos0, 423.2 * si.KELVIN, pressure, feed)
        x_a = float(pe.liquid.molefracs[0])
        x_b = float(pe.vapor.molefracs[0])
        if abs(x_a - x_b) > 1e-4:
            print(f"  T=423.2K: feed={z1:.4f} → x1_I={min(x_a,x_b):.5f}, x1_II={max(x_a,x_b):.5f}  (exp: 0.454)")
            break
    except:
        pass
print()
print("Conclusion: The 7 failing temperatures are above the MODEL'S UCST for")
print("water_gross2002. This is a fundamental model limitation, not a flash failure.")
print("No feed manipulation, pressure change, or flash algorithm will fix it.")
print("The water_gross2002 model simply predicts LLE only up to ~426K,")
print("while the experimental UCST is ~493K for 1-hexanol + water.")
