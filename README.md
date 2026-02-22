# fit-pcsaft

Fit PC(P)-SAFT pure component parameters to experimental vapor pressure and liquid density data, wrapping [FeOs](https://github.com/feos-org/feos) for equation of state evaluation and automatic differentiation.

## Installation

```bash
pip install git+https://github.com/maximilianfv/fit-pcsaft.git
```

## Quick start

```python
from fit_pcsaft import fit_pure

result = fit_pure(
    id="ethanol",
    psat_path="data/psat/ethanol.csv",
    density_path="data/density/ethanol.csv",
    na=1, nb=1,  # 2B association scheme
)
print(result)
result.to_json("parameters.json")
result.plot(path="ethanol.png")
```

```
Fitted parameters:
  m (segments):            3.6856
  σ (diameter):            2.7184 Å
  ε/k (energy):            175.51 K
  κ_ab (assoc. volume):    0.111161
  ε_ab/k (assoc. energy):  2182.12 K

Association scheme:        2B (na=1, nb=1)

Fitting quality:
  ARD vapor pressure:      6.19%  (n=458)
  ARD liquid density:      0.23%  (n=161)
  RMS weighted resid.:     0.0009
  Converged:               True
  Function evals:          20
  Time elapsed:            1.66 s
```

## Data format

CSV with temperature in the first column and the property in the second. Extra columns are ignored — NIST TDE exports work directly.

```
Temperature ( K ),vapor pressure ( kPa ),...
231.15,100.0,...
```

Default units: **K**, **kPa**, **kg/m³**.

## Examples

### Non-associating — propane

```python
fit_pure(id="propane", psat_path=..., density_path=...)
```

### Polar — acetone (fit dipole moment)

```python
fit_pure(id="acetone", psat_path=..., density_path=..., mu=None)
```

### Associating — ethanol (2B scheme)

```python
fit_pure(id="ethanol", psat_path=..., density_path=..., na=1, nb=1)
```

### HBD/HBA metadata only — e.g. carbonyl oxygen

Passes `na`/`nb` to the output JSON without fitting association parameters:

```python
fit_pure(id="acetone", psat_path=..., density_path=..., na=1)  # nb defaults to 0
```

### Quadrupolar — CO₂

```python
fit_pure(id="carbon dioxide", psat_path=..., density_path=..., q=4.4)
# Note: q != 0 triggers numerical Jacobian (~3 min)
```

### Robust loss — outlier-heavy data

```python
fit_pure(..., loss="huber", f_scale=0.01)
```

## API reference

### `fit_pure`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | — | Compound name, SMILES, or InChI (resolved via PubChem) |
| `psat_path` | `str\|Path` | — | Path to vapor pressure CSV |
| `density_path` | `str\|Path` | — | Path to liquid density CSV |
| `mu` | `float\|None` | `0.0` | Dipole moment (D). `None` to fit it |
| `q` | `float` | `0.0` | Quadrupole moment (D·Å), fixed. Triggers numerical Jacobian |
| `na` | `int\|None` | `None` | Association sites A. `na>0, nb>0` → fit κ_ab and ε_ab/k |
| `nb` | `int\|None` | `None` | Association sites B. `nb=0` → metadata only, no assoc. fitting |
| `psat_weight` | `float` | `3.0` | Relative weight of vapor pressure residuals |
| `density_weight` | `float` | `2.0` | Relative weight of density residuals |
| `extrapolate_psat` | `bool` | `False` | Clausius-Clapeyron extrapolation for near-critical/supercritical data points |
| `loss` | `str` | `'linear'` | Scipy loss function: `'linear'`, `'huber'`, `'soft_l1'`, `'cauchy'`, `'arctan'`. Non-linear losses switch solver to TRF |
| `f_scale` | `float` | `1.0` | Soft margin for robust loss functions |
| `temperature_unit` | `si.SIObject` | `si.KELVIN` | Temperature unit in CSV |
| `pressure_unit` | `si.SIObject` | `si.KILO * si.PASCAL` | Pressure unit in CSV |
| `density_unit` | `si.SIObject` | `si.KILOGRAM / si.METER**3` | Density unit in CSV |
| `scipy_kwargs` | `dict\|None` | `None` | Override `scipy.least_squares` defaults |

### `FitResult`

| Attribute | Description |
|---|---|
| `params` | Dict of fitted parameters (`m`, `sigma`, `epsilon_k`, optionally `mu`, `kappa_ab`, `epsilon_k_ab`) |
| `eos` | Ready-to-use `feos.EquationOfState` |
| `ard_psat` | ARD% for vapor pressure |
| `ard_rho` | ARD% for liquid density |
| `time_elapsed` | Wall time for the optimization loop (s) |
| `scipy_result` | Raw `scipy.OptimizeResult` |

#### `result.to_json(path)`

Appends the fitted record to a feos-compatible JSON parameter file. Creates the file and parent directories if needed.

#### `result.plot(path=None)`

Two-panel phase diagram (Clausius-Clapeyron + T-ρ) with experimental data overlay. Returns `(fig, axes)`. Saves to `path` if provided.
