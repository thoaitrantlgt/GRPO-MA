# Incomplete execution

All455 count allocations were evaluated, but the first execution failed during AUDIT.json serialization because a NumPy bool comparison is not a native JSON boolean. No summary, raw lattice or decision was written. The correction converts audit checks to native booleans; numerical code, benchmark, geometries, estimators, thresholds and bootstrap protocol are unchanged. Completed execution uses20261011_run02; this directory and its pre-outcome snapshots are preserved.
