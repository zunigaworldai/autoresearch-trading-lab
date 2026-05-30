#!/usr/bin/env bash
set -euo pipefail

echo "=== Karpathy AutoResearch Trading Audit ==="
python scripts/research/autoresearch_audit.py

echo ""
echo "Generated files:"
ls -lh outputs/research/autoresearch/latest_research_audit.*
