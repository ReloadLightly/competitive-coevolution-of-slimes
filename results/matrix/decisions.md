# Decision log for the experiment matrix

Anything that changed after the protocol was fixed is recorded here, with what
was known at the time.

## 2026-08-18 14:41 UTC — protocol fixed, matrix launched

27 runs: control 6 seeds, hof-0.25 6, hof-0.50 3, sigma-0.05 3, sigma-0.20 3,
pop-32 3, pop-512 3. Protocol in `protocol.json`. Budget was set from an
estimated ~30 min per run.

## 2026-08-18 15:00 UTC — seed count increased

Runs turned out to take ~17.3 min rather than ~30, leaving spare capacity.
The seed count was therefore raised to 12 for `control` and `hof-0.25` and to 6
for the four side conditions.

State of knowledge at the time of this decision: exactly one run of the matrix
had finished (`control_s103`: final -0.29, peak +0.17). No hall-of-fame run had
completed, no condition had been compared with another, and
`analyze_matrix.py` had not been run on any real data. The change is therefore
a budget decision, not a data-dependent one.

Seeds 101-106 (all conditions) come from the first wave; 107-112 (control,
hof-0.25) and 104-106 (side conditions) come from the second. All are reported
together; none is dropped.
