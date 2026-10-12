#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

# Every stage is idempotent and refuses ambiguous partial outputs.
bash scripts/physical_tokens/run_d28_export_full_spatiotemporal_cache.sh
bash scripts/physical_tokens/run_d28_validate_and_sidecar.sh
bash scripts/physical_tokens/run_d29_full_diversity_codebook_sweep.sh
bash scripts/physical_tokens/run_d29_evaluate_codebook_sweep.sh
bash scripts/physical_tokens/run_d30_selected_oracle_conditioned_gate.sh
bash scripts/physical_tokens/run_d31_full_diversity_dual_token_probe.sh
