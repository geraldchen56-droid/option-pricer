#!/bin/bash
# Double-click this file in Finder to run the project's demos.

# A double-clicked .command starts in your home folder, so move to this
# file's own folder first.
cd "$(dirname "$0")"

# Full path to python, because Finder does not load your shell's PATH.
PY=/opt/anaconda3/bin/python3

echo "=== Black-Scholes price and Greeks ==="
$PY -m option_pricer.cli -S 100 -K 100 -T 1 -r 5% -v 20%

echo
echo "=== Monte Carlo call price vs Black-Scholes ==="
$PY -m option_pricer.monte_carlo

echo
echo "=== Delta hedging a short call ==="
$PY -m option_pricer.delta_hedge

echo
echo "Done. Close this window when you are finished."
