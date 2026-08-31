#!/usr/bin/env bash
set -o pipefail

pytest /tests/elasticity_test.py
status=$?

if [ "$status" -eq 0 ]; then
    echo "1" > /logs/verifier/reward.txt
else
    echo "0" > /logs/verifier/reward.txt
fi

exit "$status"
