"""Backward-compatible entrypoint for inference.

Prefer importing or executing ts_type_refiner.inference.infer directly.
"""

from ts_type_refiner.inference.infer import main


if __name__ == "__main__":
    main()
