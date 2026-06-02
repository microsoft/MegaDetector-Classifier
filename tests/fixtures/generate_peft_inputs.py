"""Regen script for PEFT-disabled behavior-snapshot inputs (consumed by P7).

Run this script to regenerate tests/fixtures/peft_disabled_{input,target}.pt
from a fixed seed. Idempotent: rerunning produces byte-identical files.

NOTE: tests/fixtures/peft_disabled_golden.pt is INTENTIONALLY NOT generated
here — it depends on DINOv3Linear (P4) and is produced by
tests/regression/regenerate_peft_disabled_golden.py in P7.
"""
from pathlib import Path

import torch

BATCH = 4
IMAGE_SIZE = 518
NUM_CLASSES = 4
SEED = 42

FIXTURES_DIR = Path(__file__).resolve().parent


def main() -> None:
    g_x = torch.Generator().manual_seed(SEED)
    g_y = torch.Generator().manual_seed(SEED + 1)
    x = torch.empty(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE).normal_(generator=g_x)
    y = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH,), generator=g_y, dtype=torch.long)
    torch.save(x, FIXTURES_DIR / "peft_disabled_input.pt")
    torch.save(y, FIXTURES_DIR / "peft_disabled_target.pt")
    print(f"Wrote peft_disabled_input.pt shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"Wrote peft_disabled_target.pt shape={tuple(y.shape)} dtype={y.dtype}")


if __name__ == "__main__":
    main()
