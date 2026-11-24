#!/usr/bin/env bash
# Deploy this repository to a Hugging Face Docker Space.
# Usage: scripts/deploy_hf_space.sh <hf-username>/<space-name>
# Requires: `hf auth login` (a token with write access).
set -euo pipefail
SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
cd "$(dirname "$0")/.."
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# Copy the tracked files only (no .venv, node_modules, runs, ...).
git ls-files -z | rsync -a --from0 --files-from=- ./ "$STAGE/"
# A Space's README must start with YAML front matter that selects the Docker SDK.
{
  printf -- '---\ntitle: PINN Equation Solver\nemoji: 📈\ncolorFrom: blue\ncolorTo: indigo\nsdk: docker\napp_port: 7860\npinned: false\nshort_description: Validation-driven PINN solver for ODEs and 1D parabolic PDEs\n---\n\n'
  cat README.md
} > "$STAGE/README.md"
rm -rf "$STAGE/docs" "$STAGE/benchmark_results" "$STAGE"/*.log   # not needed by the app; keeps the Space small
hf repo create "$SPACE" --repo-type space --space-sdk docker --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy from $(git rev-parse --short HEAD)"
echo "Space: https://huggingface.co/spaces/$SPACE"
echo "App:   https://$(echo "$SPACE" | tr '/' '-' | tr '[:upper:]_.' '[:lower:]--').hf.space"
