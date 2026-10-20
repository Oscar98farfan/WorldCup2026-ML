"""
Punto de entrada del proyecto.

    python main.py all                 # pipeline completo
    python main.py build               # integra datos y crea features
    python main.py train               # walk-forward + modelo final
    python main.py simulate --n 20000  # Monte Carlo con el cuadro oficial
    python main.py evaluate            # compara con el Mundial 2026 real
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wc import config, pipeline  # noqa: E402

STEPS = {
    "build": pipeline.step_build,
    "train": pipeline.step_train,
    "simulate": pipeline.step_simulate,
    "evaluate": pipeline.step_evaluate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldCup2026-ML")
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--n", type=int, default=config.N_SIMULATIONS,
                        help="número de simulaciones Monte Carlo")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    steps = list(STEPS) if args.step == "all" else [args.step]
    for name in steps:
        if name == "simulate":
            STEPS[name](n=args.n)
        else:
            STEPS[name]()


if __name__ == "__main__":
    main()
