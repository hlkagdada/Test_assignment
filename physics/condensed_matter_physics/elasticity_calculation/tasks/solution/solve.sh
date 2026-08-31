#!/usr/bin/env bash
set -eo pipefail

python3 /app/solution/elastic_pipeline.py
python3 /app/solution/run_pipeline_summary.py
