# fit-pcsaft

A simple Python tool for fitting PC-SAFT (and PCP-SAFT) equation
of state parameters to experimental data. Built on the PC-SAFT
implementation provided by
[FeOs](https://github.com/feos-org/feos).

Setting up fitting procedures repeatedly in ad hoc notebooks is
tedious. This tool streamlines the process: provide a component
identifier and get back the fitted result along with a phase
diagram to visually assess fit quality (not data quality!).

## Installation

Install the package directly from GitHub using your preferred python package manager (e.g. `pip`):

```bash
pip install git+https://github.com/maximilianfv/fit-pcsaft.git
```

## Quick Start

Fitting parameters for a compound like **ethanol** only takes a few lines of code.

```python
from fit_pcsaft import fit_pure

# 1. Fit parameters to experimental data
result = fit_pure(
    id="ethanol",                     # Search PubChem by name, SMILES, or InChI
    psat_path="data/psat/ethanol.csv", # Vapor pressure data
    density_path="data/density/ethanol.csv", # Liquid density data
    na=1, nb=1,                        # 2B association scheme for alcohols
)

# 2. Inspect the results
print(result)

# 3. Save for future use in FeOs/other tools
result.to_json(path="my_parameters.json")

# 4. Create a phase diagram
result.plot(path="ethanol_fit.png", color="red")
```

### Example Output

```text
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
  Time elapsed:            0.26 s
```

## Data Format

The tool expects simple CSV files. The **first column** must be temperature, and the **second column** must be the property (saturation pressure or mass density).

* **Vapor Pressure**: Default units are **K** and **kPa**.
* **Liquid Density**: Default units are **K** and **kg/m³**.

## Handling Units

If your data uses different SI units (e.g., Celsius, bar, or g/cm³), you can specify them using the `si_units` package (which is installed automatically):

```python
import si_units as si

result = fit_pure(
    ...,
    temperature_unit=si.CELSIUS,
    pressure_unit=si.BAR,
    density_unit=si.GRAM / si.CENTI * si.METER**3
)
```

## Examples

For practical demonstrations of different substances (hydrocarbons, polar solvents, and associating fluids), please explore the [examples/](examples/) folder in this repo.

## Roadmap

Future features and improvements:
* [ ] **Binary Interaction Parameters**:
  * [ ] Support for **VLE** (Vapor-Liquid Equilibrium) data.
  * [ ] Support for **LLE** (Liquid-Liquid Equilibrium) data.
  * [ ] Support for **SLE** (Solid-Liquid Equilibrium) data.

* [ ] **Refactor for readability and mantainability** (this was heavily vibe coded)

## License

Licensed under either of

* [MIT](https://opensource.org/licenses/MIT)
* [Apache-2.0](https://opensource.org/licenses/Apache-2.0)
* [BSD-2-Clause](https://opensource.org/licenses/BSD-2-Clause)

at your option.
