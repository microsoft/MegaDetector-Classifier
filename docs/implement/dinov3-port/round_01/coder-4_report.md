# coder-4 docs report

## Files
- `docs/dinov3.md` — new full DINOv3 documentation covering overview, optional install, gated Hugging Face access, model-size table, inference, two-stage workflow, and troubleshooting.
- `README.md` — appended `## DINOv3 classification (experimental)` section with gated-access note, Zenodo placeholder, and link to `docs/dinov3.md`.

## Commit
- `63e0a5c7224d0c34ebafb6daf0f1447cf2cfe9bc`

## SIGNATURE
- `DINOv3 classification (experimental)`

## Decisions
- Kept DINOv3 documentation additive and markdown-only.
- Documented DINOv3 dependencies as optional so the base pipeline remains unaffected.
- Stated the Amazon DINOv3 Zenodo URLs as placeholders and directed users to pass a local checkpoint with `weights=<path>` until publication.
- Used the requested Hugging Face gated-access workflow and two-stage commands.

## Verification
- Ran `git diff --check -- README.md docs/dinov3.md` successfully before committing.
- Verified the README signature and required DINOv3 documentation anchors with `rg`.

STATUS: DONE COMMIT=63e0a5c7224d0c34ebafb6daf0f1447cf2cfe9bc SIGNATURE="DINOv3 classification (experimental)" ITEM=docs
