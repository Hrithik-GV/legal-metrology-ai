"""
YOLOv8 Detection Module
"""

from .yolo_detector import (
    YOLODetector,
    BaseDetector,
    HeuristicPackagingPanelDetector,
    detect_regions,
    annotate_detections,
)

__all__ = [
    "YOLODetector",
    "BaseDetector",
    "HeuristicPackagingPanelDetector",
    "detect_regions",
    "annotate_detections",
]
