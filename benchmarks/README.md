# fit-pcsaft benchmarks

Research sweep: 3 systems × ~27 water models × 3 fitting scenarios (LLE, VLE, VLE+LLE) × 2 k_ij polynomial orders.

For each combination, `run_test.py` fits k_ij, then cross-evaluates the fitted polynomial against the *unfitted* data type (e.g. an LLE-fitted k_ij is evaluated against VLE data). Results are written to `run_test.log`.

`dashboard.py` parses the log and generates an interactive `dashboard.html` (both are gitignored).

## Usage

```bash
uv run python benchmarks/run_test.py
uv run python benchmarks/dashboard.py
```

## Notes

- Pure PC-SAFT parameters for benchmark compounds are in `benchmarks/test_pure.json` (6 compounds not present in `examples/data/parameters/examples_pure.json`).
- Water models are loaded from `examples/data/parameters/water_models.json`.
- VLE/LLE data lives under `benchmarks/vle/` and `benchmarks/lle/` (some files extend the data in `examples/data/vle_lle/`).
- `run_test.py` intentionally imports private `fit_pcsaft._binary` internals for cross-evaluation helpers not yet exposed publicly.
