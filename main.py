"""
Dataset reliability assessment pipelines — unified entry point.

Usage
-----
    python main.py                                   # run all pipelines
    python main.py --pipeline compas                 # four-stage COMPAS trace
    python main.py --pipeline cifar10h               # CIFAR-10H annotation + processing
    python main.py --pipeline cifar10h_labeling      # annotation-count opinion analysis
    python main.py --pipeline gtsrb                  # GTSRB class-balance opinion
    python main.py --pipeline gtsrb --train          # GTSRB with CNN training
    python main.py --pipeline gtsrb --collaborative  # GTSRB with federated bias sweep
    python main.py --pipeline compas_balance         # COMPAS class-balance opinion
    python main.py --pipeline cifar10h_balance       # CIFAR-10H class-balance opinion
"""

import argparse
import sys


def _header(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}\n")


def run_compas():
    _header("COMPAS Dataset Reliability Assessment")
    from compas import run_compas_evaluation
    run_compas_evaluation()


def run_cifar10h():
    _header("CIFAR-10H Dataset Reliability Assessment")
    from cifar_10h import run_cifar10h_evaluation
    run_cifar10h_evaluation()


def run_cifar10h_labeling():
    _header("CIFAR-10H Annotation-Level Label Analysis")
    from cifar10h_labeling import run_cifar10h_label_analysis
    run_cifar10h_label_analysis()


def run_gtsrb(train: bool = False, collaborative: bool = False):
    _header("GTSRB Class-Balance Opinion Analysis")
    from gtsrb import run_gtsrb_analysis
    run_gtsrb_analysis(train=train, collaborative=collaborative)


def run_compas_balance():
    _header("COMPAS Class-Balance Opinion Analysis")
    from compas_balance import run_compas_balance_analysis
    run_compas_balance_analysis()


def run_cifar10h_balance():
    _header("CIFAR-10H Class-Balance Opinion Analysis")
    from cifar10h_balance import run_cifar10h_balance_analysis
    run_cifar10h_balance_analysis()


def run_all(train: bool = False, collaborative: bool = False):
    """Run every pipeline in sequence."""
    run_compas()
    run_cifar10h()
    run_cifar10h_labeling()
    run_gtsrb(train=train, collaborative=collaborative)
    run_compas_balance()
    run_cifar10h_balance()


def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Dataset Reliability Assessment — Subjective Logic Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline",
        choices=[
            "compas", "cifar10h", "cifar10h_labeling",
            "gtsrb", "compas_balance", "cifar10h_balance", "all",
        ],
        default="all",
        metavar="PIPELINE",
        help=(
            "Pipeline to run: compas | cifar10h | cifar10h_labeling | "
            "gtsrb | compas_balance | cifar10h_balance | all  (default: all)"
        ),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="(gtsrb) Train the CNN before running the opinion analysis.",
    )
    parser.add_argument(
        "--collaborative",
        action="store_true",
        help="(gtsrb) Run the federated bias sweep.",
    )

    args = parser.parse_args()

    try:
        if args.pipeline == "compas":
            run_compas()
        elif args.pipeline == "cifar10h":
            run_cifar10h()
        elif args.pipeline == "cifar10h_labeling":
            run_cifar10h_labeling()
        elif args.pipeline == "gtsrb":
            run_gtsrb(train=args.train, collaborative=args.collaborative)
        elif args.pipeline == "compas_balance":
            run_compas_balance()
        elif args.pipeline == "cifar10h_balance":
            run_cifar10h_balance()
        else:
            run_all(train=args.train, collaborative=args.collaborative)

    except FileNotFoundError as exc:
        print(f"\n[Error] Data file not found: {exc}", file=sys.stderr)
        print("Ensure required CSV / NPY files are present in ./data/", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[Error] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
