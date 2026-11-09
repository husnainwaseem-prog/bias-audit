"""CLI for the Model Explainability and Bias Audit.

Usage:
    python -m src.cli audit [--data-dir data] [--out-dir results] [--shap-n 2000]
    python -m src.cli explain --row 0 [--data-dir data] [--out results/plots]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    p = argparse.ArgumentParser(description="Model Explainability and Bias Audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="Run the full audit (3 models + SHAP + fairness)")
    a.add_argument("--data-dir", default="data")
    a.add_argument("--out-dir", default="results")
    a.add_argument("--shap-n", type=int, default=100)

    args = p.parse_args()

    if args.cmd == "audit":
        from src.audit import run_audit
        run_audit(data_dir=args.data_dir, out_dir=args.out_dir, shap_n=args.shap_n)


if __name__ == "__main__":
    main()
