"""Final check: phase labeling in PhaseDiagram.lle and the overall model behavior.

The PD.lle output labels phases as 'liquid' and 'vapor', but both are liquid phases.
Which is phase I (water-lean) and which is phase II (hexanol-rich)?

Also verify the tp_flash behavior for k_ij=0.1 at low T — the composition
predictions show x1_I ~ 0.1 (much higher than exp 0.001). This is a MODEL
issue, not a flash failure.
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
print("TEST 1: PhaseDiagram.lle phase labeling")
print("=  'liquid' = low-x1 phase, 'vapor' = high-x1 phase OR vice versa?")
print("=" * 70)

eos_k = build_eos(0.1)
feed = np.array([0.10, 0.90]) * si.MOL
pd = feos.PhaseDiagram.lle(
    eos_k, pressure, feed,
    min_tp=270.0 * si.KELVIN,
    max_tp=500.0 * si.KELVIN,
    npoints=300
)
T_pd = pd.liquid.temperature / si.KELVIN
xI_liq = pd.liquid.molefracs[:, 0]   # x1 in 'liquid' phase
xI_vap = pd.vapor.molefracs[:, 0]    # x1 in 'vapor' phase

print("Sample of PD.lle output (k_ij=0.1, feed=0.10):")
print(f"{'T(K)':>8} {'liq.x1':>10} {'vap.x1':>10}  note")
for i in range(0, len(T_pd), max(1, len(T_pd)//10)):
    note = ""
    if xI_liq[i] > xI_vap[i]:
        note = "<-- liq is HIGH x1 phase"
    else:
        note = "<-- liq is LOW x1 phase"
    print(f"{T_pd[i]:8.1f} {xI_liq[i]:10.6f} {xI_vap[i]:10.6f}  {note}")

print()
print("Conclusion on phase labeling:")
print(f"  At T={T_pd[0]:.1f}K: liquid.x1={xI_liq[0]:.6f}, vapor.x1={xI_vap[0]:.6f}")
if xI_liq[0] > xI_vap[0]:
    print("  'liquid' = HIGH x1 (hexanol-rich) = phase II")
    print("  'vapor'  = LOW x1 (water-rich)    = phase I")
else:
    print("  'liquid' = LOW x1 (water-rich) = phase I")
    print("  'vapor'  = HIGH x1 (hexanol-rich) = phase II")

print()
print("=" * 70)
print("TEST 2: Water_gross2002 fundamental limitation summary")
print("=" * 70)
print()
print("The water_gross2002 model (2B, na=nb=1) with PC-SAFT predicts:")
print()

for kij in [-0.2, -0.1, 0.0, 0.1, 0.2]:
    eos_k = build_eos(kij)
    # Find max T where LLE exists (UCST)
    ucst = None
    for feed_z1 in [0.10, 0.15, 0.20, 0.30, 0.40]:
        try:
            feed = np.array([feed_z1, 1.0 - feed_z1]) * si.MOL
            pd = feos.PhaseDiagram.lle(
                eos_k, pressure, feed,
                min_tp=270.0 * si.KELVIN,
                max_tp=500.0 * si.KELVIN,
                npoints=200
            )
            T = pd.liquid.temperature / si.KELVIN
            if len(T) > 0:
                if ucst is None or T.max() > ucst:
                    ucst = T.max()
        except:
            pass
    if ucst:
        print(f"  k_ij={kij:+.3f}: UCST ~ {ucst:.1f}K  (exp: 493K)")
    else:
        print(f"  k_ij={kij:+.3f}: No LLE found")

print()
print("NOTE: The experimental UCST for 1-hexanol + water is ~493K.")
print("All k_ij values tested give UCST < 430K with water_gross2002.")
print("This is a fundamental model limitation — the 2B water model grossly")
print("underestimates the LLE mutual miscibility and UCST temperature.")
print()
print("The 7 failing temperatures (433-493K) CANNOT be recovered by any")
print("flash algorithm — the model simply predicts single-phase behavior there.")
print()
print("=" * 70)
print("TEST 3: tp_flash robustness assessment for optimization")
print("=" * 70)
print()
print("For the working range (273-423K), how reliable is tp_flash with")
print("the current 51 sigmoid feeds?")
print()

_i = np.arange(-50, 51, 2, dtype=float)
_s = 0.05 * _i + 6e-5 * _i**3
SIGMOID_FEEDS = (1.0 / (1.0 + np.exp(_s))).tolist()

# Test at 3 k_ij values × 10 temperatures
T_test = [273.2, 283.2, 298.2, 323.2, 343.2, 363.2, 383.2, 393.2, 403.2, 413.2]
for kij in [0.0, 0.1, 0.2]:
    eos_k = build_eos(kij)
    successes = 0
    for T_K in T_test:
        for z1 in SIGMOID_FEEDS:
            try:
                feed = np.array([z1, 1.0 - z1]) * si.MOL
                pe = feos.PhaseEquilibrium.tp_flash(eos_k, T_K * si.KELVIN, pressure, feed)
                x_a = float(pe.liquid.molefracs[0])
                x_b = float(pe.vapor.molefracs[0])
                if abs(x_a - x_b) > 1e-4:
                    successes += 1
                    break
            except:
                pass
    print(f"k_ij={kij:+.2f}: {successes}/{len(T_test)} successful at sigmoid feeds only")

# Test same with extra feeds
extra = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
for kij in [0.0, 0.1, 0.2]:
    eos_k = build_eos(kij)
    successes = 0
    for T_K in T_test:
        for z1 in extra + SIGMOID_FEEDS:
            z1 = float(np.clip(z1, 1e-4, 1-1e-4))
            try:
                feed = np.array([z1, 1.0 - z1]) * si.MOL
                pe = feos.PhaseEquilibrium.tp_flash(eos_k, T_K * si.KELVIN, pressure, feed)
                x_a = float(pe.liquid.molefracs[0])
                x_b = float(pe.vapor.molefracs[0])
                if abs(x_a - x_b) > 1e-4:
                    successes += 1
                    break
            except:
                pass
    print(f"k_ij={kij:+.2f}: {successes}/{len(T_test)} with extra + sigmoid feeds")
