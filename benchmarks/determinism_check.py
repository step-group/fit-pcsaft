"""SBX vs DE reproducibility at identical settings, same seed, twice each."""
from pathlib import Path
import numpy as np
from fit_pcsaft import fit_pure_pareto

DATA = Path.home() / "projects" / "fit-pcsaft-knobs" / "examples" / "data"
BASE = dict(
    id="water",
    psat_path=DATA / "psat" / "water.csv",
    density_path=DATA / "density" / "water.csv",
    sft_path=DATA / "surface_tension" / "water.csv",
    na=1, nb=1, objectives=("vle", "sft"),
    bounds=[(0.8,3.0),(2.0,3.5),(150.0,400.0),(1e-3,0.35),(1500.0,4000.0)],
    pop_size=30, n_gen=30, n_restarts=1, refine=0, seed=101, verbose=False,
)

def sig(f):
    F = np.asarray(f.F)
    return (F.shape[0], round(float(F[:,0].min()),6), round(float(F[:,1].min()),6))

def main():
    for label, over in (("sbx", {}), ("de", dict(variant="de", de_cr=1.0, de_f=0.5))):
        s = [sig(fit_pure_pareto(**BASE, **over)) for _ in range(3)]
        same = all(x == s[0] for x in s)
        print(f"{label:>4}: {'REPRODUCIBLE' if same else 'DIVERGED'}  {s}")

if __name__ == "__main__":
    main()
