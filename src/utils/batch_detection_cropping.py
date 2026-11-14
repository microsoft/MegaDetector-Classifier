# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Batch MegaDetector detection and cropping, backed by sparrow-engine."""

import os

import numpy as np
import pandas as pd
import supervision as sv
import sparrow_engine

from src.utils import utils

DEFAULT_DETECTOR = "MDV6-yolov10-e"
DEFAULT_THRESHOLD = 0.2


def _ensure_model(model):
    """Download the detector from the Sparrow model zoo if it is not installed."""
    if model not in {m.id for m in sparrow_engine.list_models()}:
        sparrow_engine.download_model(model)


def _list_annotated_images(folder_path, output_path, annotation_file):
    """Images under folder_path whose file name appears in the annotation CSV.

    The crop output folder is skipped so crops from an earlier split are not re-detected.
    """
    wanted = set(pd.read_csv(annotation_file)['path'].astype(str).map(os.path.basename))
    output_path = os.path.abspath(output_path)
    paths = []
    for dirpath, dirnames, filenames in os.walk(folder_path):
        dirnames[:] = [d for d in dirnames if os.path.abspath(os.path.join(dirpath, d)) != output_path]
        paths.extend(os.path.join(dirpath, f) for f in filenames if f in wanted)
    return sorted(paths)


def to_supervision(result):
    """Convert a sparrow_engine DetectResult into the {img_id, detections} dict save_crop_images expects."""
    width, height = result.image_size
    dets = result.detections
    detections = sv.Detections(
        xyxy=np.array([d.bbox.to_pixels(width, height) for d in dets], dtype=np.float32).reshape(-1, 4),
        confidence=np.array([d.confidence for d in dets], dtype=np.float32),
        class_id=np.array([d.label_id for d in dets], dtype=int),
    )
    return {"img_id": result.path, "detections": detections}


def batch_detection_cropping(folder_path, output_path, annotation_file,
                             model=DEFAULT_DETECTOR, threshold=DEFAULT_THRESHOLD):
    """Detect animals/people/vehicles with MegaDetector and save one crop per detection.

    Args:
        folder_path (str): Root folder of the images (searched recursively).
        output_path (str): Folder for the cropped images.
        annotation_file (str): CSV with `path`, `classification` and `label` columns.
        model (str): sparrow-engine detector id, e.g. "MDV6-yolov10-e".
        threshold (float): Detection confidence threshold.
    Returns:
        str: Path to the CSV describing the crops.
    """
    _ensure_model(model)
    paths = _list_annotated_images(folder_path, output_path, annotation_file)
    results = sparrow_engine.detect(paths, model=model, threshold=threshold) if paths else []
    return utils.save_crop_images([to_supervision(r) for r in results], output_path, annotation_file)
