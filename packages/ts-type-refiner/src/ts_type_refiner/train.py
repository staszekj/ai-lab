"""Backward-compatible entrypoint for training.

Prefer importing or executing ts_type_refiner.training.train directly.
"""

from ts_type_refiner.training.train import main


if __name__ == "__main__":
    main()
