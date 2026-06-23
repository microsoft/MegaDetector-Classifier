# Inquisitor approvals — round 2

Bar this round is HIGH (confirming round; round-1 verified 13/13, NEW=0, 56 tests pass).
Approve ONLY genuine correctness fixes (real bug / edge case / plan-spec violation) that are
minimal and do not risk §12 import invariants, SINCHI transform parity, or the 56 passing tests.
Each item was opened at its cited location and judged independently against
DINOv3_classification_port_plan.md (§6, §9, §11, §12). Auditor section is NOTHING-TO-DO.

## Auditor verdicts
(NOTHING-TO-DO — no items. The round-2 auditor owns the 6 model files, found no structural
work above the dedup bar, and confirmed §12 import gating and SINCHI parity intact. Concur.)

## Reviewer verdicts

ITEM-REV-001 | APPROVED — Genuine silent-corruption bug AND an explicit plan §9.1 violation.
  extract_embeddings.py:177 `if embedding_path.exists(): continue` silently reuses the FIRST
  image's embedding whenever two distinct `Photo_ID`s collapse to the same `safe_name`
  (`/`,`\`→`_`), so a different image's CSV row is later served the wrong cached vector.
  Plan §9.1 explicitly prescribes the chosen remedy verbatim: "write a manifest.csv
  (image_path,pkl_path,label) and detect/raise on collisions." The proposed guard
  (raise on two different paths → same pkl; verify stored `image_path` on resume-skip;
  write manifest) is exactly that and is minimal. No test exercises this script, so no
  regression to the 56. Keep the resume-skip path intact (same image → skip; mismatch → raise).

ITEM-REV-002 | REJECTED: over-engineering + fixes a dead path — `DINOv3EmbeddingDataModule` has
  ZERO references anywhere in src/scripts/tests/configs (grep-confirmed); the documented stage-2
  entrypoint uses `DINOv3EmbeddingDataset` directly, not the DataModule. So the C3 fix here lands
  on an unexercised code path. Worse, the item bundles a speculative public-API refactor
  ("a shared helper that can build one stable mapping from one or more CSV/DataFrame inputs" +
  "make build_label_mapping() delegate to it") that touches the tested `build_label_mapping`
  (used by `DINOv3EmbeddingDataset`) and adds coupling for marginal gain — the kind of structural
  expansion this confirming round's HIGH bar exists to filter. The genuine C3 correctness fix
  belongs in the actually-run CLI path and is covered by ITEM-REV-003; that fix needs only a small
  inline union, not this refactor. (A minimal DataModule mirror could be revisited separately if the
  DataModule ever becomes a real consumer, but not as this bundled refactor.)

ITEM-REV-003 | APPROVED with MODIFY — This is the real C3 fix on the documented path. Running
  train_dinov3.py with a `val_csv` containing a class absent from `train_csv` crashes:
  the val `DINOv3EmbeddingDataset` raises KeyError at dinov3_embedding.py:94
  ("Label ... is not in label_to_idx") because `label_to_idx` is built train-only (line 54).
  Building the mapping from the union of train+val (sorted-unique), deriving
  num_classes/class_names from it, passing one shared mapping to both datasets, and comparing the
  `num_classes` override against the union size, is correct and minimal. In the common case
  (val ⊆ train) behavior is unchanged; only the val-only-class edge changes crash→trains-with-
  extra-output. No §12/import or SINCHI-transform impact (label space only); no test touches this
  script. MODIFY: build the union INLINE in train_dinov3.py (e.g. concat the two CSVs' label columns
  → sorted-unique) or by merging two `build_label_mapping` calls; do NOT introduce the rejected
  REV-002 shared-helper refactor. Keep the fix self-contained to the script.

ITEM-REV-004 | REJECTED: polish/consistency with real regression risk and ~zero correctness value.
  The item is explicitly framed as dedup ("should not maintain near-identical label-index
  builders"), not a correctness fix. The two builders DIFFER ON PURPOSE: stage-1
  `_label_mapping` returns `{}` when the Label column is absent so unlabeled/predict extraction
  yields `label=-1` (a §9.1-required feature), whereas the shared `build_label_mapping` RAISES on a
  missing label column. Swapping in the shared helper risks breaking §9.1 unlabeled extraction.
  Moreover the divergence is harmless for the documented training path: stage-2 ignores the pickle's
  stored `label` and recomputes labels from the CSV via `label_to_idx` in
  `DINOv3EmbeddingDataset.__getitem__`, so the stored-label computation never affects training
  correctness. Dedup for its own sake, risking a supported path, fails the HIGH bar.

ITEM-REV-005 | APPROVED — Both halves are spec-conformant and minimal. (1) Adding
  `transform="dinov3_bicubic_518_imagenet"` and `transform_version=1`: plan §6's authoritative
  classifier-only metadata block lists exactly these keys; the current export omits them. The
  flavour-D loader (base_classifier.py:147-162) validates these values when present and accepts the
  canonical pair, so emitting them brings stage-2 output into §6 conformance and carries transform
  identity through the round-trip (§14 drift mitigation) with no risk. (2) Raising ValueError when
  `model_name` is missing: the exporter unconditionally stamps `format="dinov3_classifier_only"`,
  `version=1`, which makes the loader treat it as `declared=True` and REQUIRE `model_name`
  (base_classifier.py:134-139). Emitting a declared flavour-D dict with `model_name=None` is an
  internally contradictory, unloadable checkpoint; a one-line fail-fast guard matches §6
  ("model_name authoritative/required") and converts a guaranteed load-time failure into an
  immediate, clear export-time error. Unreachable from train_dinov3.py (model_name is always set via
  resolve_model_id), and no test constructs/export-tests DINOv3LinearProbe, so no regression to the 56.

STATUS: APPROVALS-DONE
