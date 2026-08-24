#!/usr/bin/env bash
set -euo pipefail

# Fast public checks and a synthetic end-to-end application twin.
python -m unittest discover -s tests -v
python -m examples.dagguard_quickstart
python synthetic_application_twin.py --out results/synthetic_application_twin

# The complete simulation suite is intentionally opt-in because it is
# computationally substantial. Run as:
#   DAGGUARD_FULL=1 bash reproduce_submission.sh
if [[ "${DAGGUARD_FULL:-0}" == "1" ]]; then
  python candidate_contamination_simulations.py \
    --replicates 100 --d 20 --n 500 \
    --out results/candidate_contamination
  python additional_noise_sensitivity.py \
    --simulation-replicates 20 \
    --out results/additional_noise_sensitivity
fi

printf '\nDAGGuard reproduction stage completed successfully.\n'
