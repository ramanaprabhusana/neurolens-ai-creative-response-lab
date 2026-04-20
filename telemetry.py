"""Developer telemetry helpers for NeuroLens AI.

The helpers in this module are intentionally stateless from a persistence
standpoint. They measure the current process, current image, and current
Streamlit cache behavior without writing to a database.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy
from sklearn.cluster import MiniBatchKMeans

from analytics import (
    ACTION_LABEL_WEIGHTS,
    COLOR_MAX_SIDE,
    COLOR_SAMPLE_PIXELS,
    ENTROPY_MAX_SIDE,
    HEATMAP_MAX_SIDE,
    ColorInsight,
    clutter_score,
    nearest_color_psychology,
    resize_for_analysis,
    sample_color_pixels,
)


_CACHE_MISS_COUNTS: dict[str, int] = {}


@dataclass
class TelemetryRecorder:
    """Collect per-function latency in milliseconds."""

    timings_ms: dict[str, float] = field(default_factory=dict)

    def add(self, label: str, seconds: float) -> None:
        self.timings_ms[label] = round(seconds * 1000, 2)


@contextmanager
def execution_timer(label: str, recorder: TelemetryRecorder) -> Iterator[None]:
    """Measure an operation and store its latency in milliseconds."""
    start = time.perf_counter()
    try:
        yield
    finally:
        recorder.add(label, time.perf_counter() - start)


def begin_cache_probe(label: str) -> int:
    """Capture the current miss count before calling a cached function."""
    return _CACHE_MISS_COUNTS.get(label, 0)


def mark_cache_miss(label: str) -> None:
    """Mark that Streamlit executed the cached function body."""
    _CACHE_MISS_COUNTS[label] = _CACHE_MISS_COUNTS.get(label, 0) + 1


def end_cache_probe(label: str, previous_count: int) -> str:
    """Return Hit/Miss by checking whether the cached body executed."""
    return "Miss" if _CACHE_MISS_COUNTS.get(label, 0) > previous_count else "Hit"


def image_memory_mb(image: Image.Image | np.ndarray) -> float:
    """Calculate the in-memory footprint of the normalized image array."""
    image_rgb = _as_rgb_array(image)
    return round(float(image_rgb.nbytes / (1024 * 1024)), 2)


def score_ad_with_telemetry(image: Image.Image | np.ndarray) -> tuple[dict, dict[str, float], float]:
    """Score an ad while timing entropy, K-Means, and saliency components."""
    recorder = TelemetryRecorder()
    entropy = calculate_entropy_with_telemetry(image, recorder)
    clutter = clutter_score(entropy)
    colors = extract_dominant_colors_with_telemetry(image, recorder)
    focus_score = saliency_strength_with_telemetry(image, recorder)
    emotion_score = sum(
        ACTION_LABEL_WEIGHTS.get(color.psychology, 0) * (color.percentage / 100)
        for color in colors
    )
    emotion_score = float(np.clip(emotion_score * 5, 0, 100))
    final_score = (100 - clutter) * 0.45 + emotion_score * 0.35 + focus_score * 0.20
    return (
        {
            "entropy": float(entropy),
            "clutter": int(clutter),
            "colors": colors,
            "emotion_score": round(emotion_score, 1),
            "focus_score": round(focus_score, 1),
            "final_score": round(float(final_score), 1),
        },
        recorder.timings_ms,
        image_memory_mb(image),
    )


def calculate_entropy_with_telemetry(image: Image.Image | np.ndarray, recorder: TelemetryRecorder) -> float:
    """Time the Scikit-Image Shannon entropy call separately."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=ENTROPY_MAX_SIDE)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    with execution_timer("scikit_entropy_ms", recorder):
        entropy = shannon_entropy(gray)
    return float(entropy)


def extract_dominant_colors_with_telemetry(
    image: Image.Image | np.ndarray,
    recorder: TelemetryRecorder,
    k: int = 3,
) -> list[ColorInsight]:
    """Extract dominant colors while timing the K-Means clustering step."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=COLOR_MAX_SIDE)
    pixels = sample_color_pixels(image_rgb, max_pixels=COLOR_SAMPLE_PIXELS)

    unique_count = len(np.unique(pixels.astype(np.uint8), axis=0))
    cluster_count = max(1, min(k, unique_count))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=3,
        max_iter=60,
        batch_size=1024,
    )
    with execution_timer("kmeans_clustering_ms", recorder):
        labels = kmeans.fit_predict(pixels)

    counts = np.bincount(labels, minlength=cluster_count)
    total = counts.sum() or 1
    ordered = np.argsort(counts)[::-1]
    insights: list[ColorInsight] = []
    for idx in ordered[:k]:
        rgb = np.clip(kmeans.cluster_centers_[idx], 0, 255).astype(int)
        hex_code = _rgb_to_hex(rgb)
        insights.append(
            ColorInsight(
                hex=hex_code,
                percentage=float((counts[idx] / total) * 100),
                psychology=nearest_color_psychology(hex_code),
            )
        )
    return insights


def saliency_strength_with_telemetry(image: Image.Image | np.ndarray, recorder: TelemetryRecorder) -> float:
    """Estimate central saliency while timing OpenCV edge detection."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=500)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    with execution_timer("opencv_edge_detection_ms", recorder):
        edges = cv2.Canny(gray, 60, 150)
    with execution_timer("opencv_saliency_blur_ms", recorder):
        saliency = cv2.GaussianBlur(edges, (0, 0), sigmaX=9, sigmaY=9)
        saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    h, w = saliency.shape
    center = saliency[h // 4 : (h * 3) // 4, w // 4 : (w * 3) // 4]
    full_mean = float(np.mean(saliency)) + 1e-6
    center_lift = float(np.mean(center)) / full_mean
    return float(np.clip(center_lift * 50, 0, 100))


def generate_attention_heatmap_layers_with_telemetry(
    image: Image.Image | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], float]:
    """Create heavy saliency layers while timing OpenCV edge/contrast work."""
    recorder = TelemetryRecorder()
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=HEATMAP_MAX_SIDE)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    with execution_timer("heatmap_opencv_edge_detection_ms", recorder):
        edges = cv2.Canny(gray, threshold1=60, threshold2=150)
    with execution_timer("heatmap_opencv_contrast_ms", recorder):
        local_contrast = cv2.Laplacian(gray, cv2.CV_16S)
        local_contrast = cv2.convertScaleAbs(local_contrast)
    with execution_timer("heatmap_layer_render_ms", recorder):
        saliency = cv2.addWeighted(edges, 0.55, local_contrast, 0.45, 0)
        saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=11, sigmaY=11)
        saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    return image_rgb, color_heatmap, recorder.timings_ms, image_memory_mb(image_rgb)


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.shape[-1] == 4:
        return cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    if array.shape[-1] != 3:
        raise ValueError("Expected an image with 1, 3, or 4 channels.")
    return np.ascontiguousarray(array.astype(np.uint8))


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
