"""Thin CLI wrappers exposed via pyproject.toml entry points."""
from __future__ import annotations

import sys

from . import evaluate, prepare, train


def prepare_cmd():
    sys.exit(prepare.main())


def train_cmd():
    sys.exit(train.main())


def eval_cmd():
    sys.exit(evaluate.main())


def predict_cmd():
    # Lightweight one-image CLI: print predictions to stdout as JSON.
    import argparse
    import json
    from pathlib import Path
    from .inference import predict_file

    p = argparse.ArgumentParser(description="Predict landmarks for a single image.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    pred = predict_file(args.checkpoint, args.image, device=args.device)
    print(json.dumps({int(k): list(v) for k, v in pred.items()}, indent=2))
