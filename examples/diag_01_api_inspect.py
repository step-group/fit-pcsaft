"""Inspect feos API for PhaseEquilibrium, PhaseDiagram, State."""
import feos
import si_units as si

print("=== dir(feos.PhaseEquilibrium) ===")
print([x for x in dir(feos.PhaseEquilibrium) if not x.startswith('__')])

print("\n=== dir(feos.PhaseDiagram) ===")
print([x for x in dir(feos.PhaseDiagram) if not x.startswith('__')])

print("\n=== dir(feos.State) ===")
print([x for x in dir(feos.State) if not x.startswith('__')])

print("\n=== help(feos.PhaseDiagram.lle) ===")
help(feos.PhaseDiagram.lle)

print("\n=== help(feos.PhaseEquilibrium.tp_flash) ===")
help(feos.PhaseEquilibrium.tp_flash)
