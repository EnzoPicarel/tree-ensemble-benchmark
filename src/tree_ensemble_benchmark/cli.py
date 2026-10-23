"""Explicit lightweight, full, CART-study, and historical-report commands."""

import argparse
import json
from pathlib import Path
from .experiments import run
from .cart_study import run_cart_study
from .reporting import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for mode in ["quick", "full"]:
        cmd = commands.add_parser(mode)
        cmd.add_argument("--output", type=Path, required=True)
        if mode == "full":
            cmd.add_argument("--data-dir", type=Path, default=Path("data"))
            cmd.add_argument("--download", action="store_true")
            cmd.add_argument(
                "--task", choices=["classification", "regression", "both"], default="both"
            )
            cmd.add_argument("--seeds", nargs="+", type=int)
    cmd = commands.add_parser("cart")
    cmd.add_argument("--output", type=Path, required=True)
    cmd = commands.add_parser("report")
    cmd.add_argument("--historical", type=Path, default=Path("results/historical"))
    cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "report":
        print(json.dumps(report(args.historical, args.output), indent=2))
    elif args.command == "cart":
        print(f"Completed {len(run_cart_study(args.output))} synthetic CART configurations")
    elif args.command == "quick":
        run("quick", args.output)
    else:
        tasks = ["classification", "regression"] if args.task == "both" else [args.task]
        run(
            "full",
            args.output,
            data_dir=args.data_dir,
            download=args.download,
            tasks=tasks,
            seeds=args.seeds,
        )


if __name__ == "__main__":
    main()
