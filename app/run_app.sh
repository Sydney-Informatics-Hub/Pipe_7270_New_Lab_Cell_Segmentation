#!/usr/bin/env bash
# Launch the cell segmentation Streamlit app.
# Usage: ./run_app.sh
set -euo pipefail
cd "$(dirname "$0")"
exec streamlit run cell_seg_app.py "$@"
