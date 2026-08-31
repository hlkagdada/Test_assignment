#!/usr/bin/env bash
set -eo pipefail

python3 /solution/elastic_pipeline.py
python3 /solution/run_pipeline_summary.py
