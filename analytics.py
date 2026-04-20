"""Image analytics for NeuroLens AI.

This module is intentionally stateless: every function accepts an image-like
input and returns computed values without reading or writing persistent data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import plotly.express as px
from PIL import Image
from skimage.measure import shannon_entropy
from sklearn.cluster import MiniBatchKMeans


COLOR_PSYCHOLOGY = {
    "#FF0000": "High Urgency",
    "#0000FF": "Trust and Stability",
    "#00AA44": "Growth and Reassurance",
    "#FFD400": "Optimism and Attention",
    "#FF7A00": "Energy and Action",
    "#7A35FF": "Premium and Imagination",
    "#111111": "Authority and Luxury",
    "#FFFFFF": "Clarity and Simplicity",
}

ACTION_LABEL_WEIGHTS = {
    "High Urgency": 18,
    "Energy and Action": 14,
    "Optimism and Attention": 10,
    "Trust and Stability": 8,
    "Growth and Reassurance": 7,
    "Premium and Imagination": 6,
    "Authority and Luxury": 5,
    "Clarity and Simplicity": 4,
}

COLOR_EMOTION_OPTIONS = [
    "Action/Urgency",
    "Trust/Stability",
    "Growth/Reassurance",
    "Premium/Aspiration",
    "Neutral/Clarity",
]

ENTROPY_MAX_SIDE = 512
COLOR_MAX_SIDE = 360
COLOR_SAMPLE_PIXELS = 8_000
HEATMAP_MAX_SIDE = 640
ACCESSIBILITY_MAX_SIDE = 640

PSYCHOLOGY_TO_EMOTION = {
    "High Urgency": "Action/Urgency",
    "Energy and Action": "Action/Urgency",
    "Optimism and Attention": "Action/Urgency",
    "Trust and Stability": "Trust/Stability",
    "Growth and Reassurance": "Growth/Reassurance",
    "Premium and Imagination": "Premium/Aspiration",
    "Authority and Luxury": "Premium/Aspiration",
    "Clarity and Simplicity": "Neutral/Clarity",
}


@dataclass(frozen=True)
class ColorInsight:
    hex: str
    percentage: float
    psychology: str


@dataclass(frozen=True)
class AccessibilityInsight:
    score: int
    wcag_status: str
    contrast_ratio: float
    risk_pixels: float
    foreground_hex: str
    background_hex: str
    recommendations: tuple[str, ...]


def load_image(uploaded_file) -> Image.Image:
    """Load a Streamlit upload as an RGB Pillow image."""
    return Image.open(uploaded_file).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert an RGB Pillow image to an RGB OpenCV-compatible array."""
    return np.array(image.convert("RGB"))


def resize_for_analysis(image_rgb: np.ndarray, max_side: int = 700) -> np.ndarray:
    """Downscale very large images for fast, repeatable analysis."""
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image has invalid dimensions.")
    largest = max(height, width)
    if largest <= max_side:
        return image_rgb
    scale = max_side / largest
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def sample_color_pixels(image_rgb: np.ndarray, max_pixels: int = COLOR_SAMPLE_PIXELS) -> np.ndarray:
    """Return a deterministic pixel sample for interactive color clustering."""
    pixels = image_rgb.reshape(-1, 3)
    if len(pixels) > max_pixels:
        rng = np.random.default_rng(42)
        pixels = pixels[rng.choice(len(pixels), size=max_pixels, replace=False)]
    return np.ascontiguousarray(pixels, dtype=np.float32)


def calculate_entropy(image: Image.Image | np.ndarray) -> float:
    """Calculate Shannon entropy over grayscale luminance."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=ENTROPY_MAX_SIDE)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(shannon_entropy(gray))


def clutter_score(entropy: float) -> int:
    """Map image entropy to a 1-100 visual clutter score.

    Natural 8-bit image entropy typically falls between 0 and 8. A denominator
    of 7.5 makes dense ads trend high while still preserving headroom.
    """
    score = round((entropy / 7.5) * 100)
    return int(np.clip(score, 1, 100))


def extract_dominant_colors(image: Image.Image | np.ndarray, k: int = 3) -> list[ColorInsight]:
    """Extract dominant colors with K-Means and annotate their psychology."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=COLOR_MAX_SIDE)
    pixels = sample_color_pixels(image_rgb)

    unique_count = len(np.unique(pixels.astype(np.uint8), axis=0))
    cluster_count = max(1, min(k, unique_count))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=3,
        max_iter=60,
        batch_size=1024,
    )
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


def nearest_color_psychology(hex_code: str) -> str:
    """Map an arbitrary hex color to the closest psychology dictionary color."""
    target = np.array(_hex_to_rgb(hex_code))
    nearest_hex = min(
        COLOR_PSYCHOLOGY,
        key=lambda candidate: float(np.linalg.norm(target - np.array(_hex_to_rgb(candidate)))),
    )
    return COLOR_PSYCHOLOGY[nearest_hex]


def generate_attention_heatmap(image: Image.Image | np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create a simulated attention heatmap overlay using contrast and edges."""
    image_rgb, color_heatmap = generate_attention_heatmap_layers(image)
    return blend_heatmap_layers(image_rgb, color_heatmap, alpha)


def generate_attention_heatmap_layers(image: Image.Image | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create heavy saliency layers once so opacity sliders only re-blend pixels."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=HEATMAP_MAX_SIDE)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, threshold1=60, threshold2=150)
    local_contrast = cv2.Laplacian(gray, cv2.CV_16S)
    local_contrast = cv2.convertScaleAbs(local_contrast)

    saliency = cv2.addWeighted(edges, 0.55, local_contrast, 0.45, 0)
    saliency = cv2.GaussianBlur(saliency, (0, 0), sigmaX=11, sigmaY=11)
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    color_heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    return image_rgb, color_heatmap


def blend_heatmap_layers(image_rgb: np.ndarray, color_heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a precomputed heatmap layer over an RGB image."""
    safe_alpha = float(np.clip(alpha, 0, 1))
    return cv2.addWeighted(image_rgb, 1 - safe_alpha, color_heatmap, safe_alpha, 0)


def saliency_strength(image: Image.Image | np.ndarray) -> float:
    """Estimate how focused the strongest attention zone is in the center."""
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=500)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    saliency = cv2.GaussianBlur(edges, (0, 0), sigmaX=9, sigmaY=9)
    saliency = cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    h, w = saliency.shape
    center = saliency[h // 4 : (h * 3) // 4, w // 4 : (w * 3) // 4]
    full_mean = float(np.mean(saliency)) + 1e-6
    center_lift = float(np.mean(center)) / full_mean
    return float(np.clip(center_lift * 50, 0, 100))


def score_ad(image: Image.Image | np.ndarray) -> dict:
    """Score a creative for the mock A/B predictor."""
    entropy = calculate_entropy(image)
    clutter = clutter_score(entropy)
    colors = extract_dominant_colors(image)
    emotion_score = sum(
        ACTION_LABEL_WEIGHTS.get(color.psychology, 0) * (color.percentage / 100)
        for color in colors
    )
    emotion_score = float(np.clip(emotion_score * 5, 0, 100))
    focus_score = saliency_strength(image)

    final_score = (100 - clutter) * 0.45 + emotion_score * 0.35 + focus_score * 0.20
    return {
        "entropy": entropy,
        "clutter": clutter,
        "colors": colors,
        "emotion_score": round(emotion_score, 1),
        "focus_score": round(focus_score, 1),
        "final_score": round(float(final_score), 1),
    }


def analyze_accessibility_contrast(
    image: Image.Image | np.ndarray,
    color_profile: list[ColorInsight] | None = None,
) -> tuple[AccessibilityInsight, np.ndarray]:
    """Estimate text/background contrast risk and render a low-contrast risk overlay.

    This is not OCR. It approximates the platform/accessibility risk by pairing
    dominant luminance clusters with edge-dense low-local-contrast regions where
    ad copy commonly lives.
    """
    image_rgb = resize_for_analysis(_as_rgb_array(image), max_side=ACCESSIBILITY_MAX_SIDE)
    colors = color_profile or extract_dominant_colors(image_rgb, k=5)
    foreground, background, ratio = _best_contrast_pair(colors)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray32 = gray.astype(np.float32)
    mean = cv2.blur(gray32, (31, 31))
    sq_mean = cv2.blur(gray32 * gray32, (31, 31))
    local_std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    soft_edges = cv2.Canny(gray, 20, 70)
    soft_edges = cv2.dilate(soft_edges, np.ones((5, 5), dtype=np.uint8), iterations=1)

    risk_mask = (soft_edges > 0) & (local_std < 28)
    risk_pixels = float(np.mean(risk_mask) * 100)

    ratio_score = np.clip((ratio / 7.0) * 100, 0, 100)
    risk_penalty = np.clip(risk_pixels * 2.2, 0, 35)
    score = int(round(np.clip(ratio_score - risk_penalty, 0, 100)))
    wcag_status = _wcag_status(ratio)
    recommendations = _accessibility_recommendations(ratio, risk_pixels)
    overlay = _accessibility_overlay(image_rgb, risk_mask)

    return (
        AccessibilityInsight(
            score=score,
            wcag_status=wcag_status,
            contrast_ratio=round(float(ratio), 2),
            risk_pixels=round(risk_pixels, 2),
            foreground_hex=foreground.hex,
            background_hex=background.hex,
            recommendations=tuple(recommendations),
        ),
        overlay,
    )


def compare_ads(image_a: Image.Image | np.ndarray, image_b: Image.Image | np.ndarray) -> dict:
    """Compare two ads with the deterministic mock predictor."""
    score_a = score_ad(image_a)
    score_b = score_ad(image_b)
    winner = "Ad A" if score_a["final_score"] >= score_b["final_score"] else "Ad B"
    margin = abs(score_a["final_score"] - score_b["final_score"])
    return {"ad_a": score_a, "ad_b": score_b, "winner": winner, "margin": round(margin, 1)}


def colors_to_plotly_rows(colors: Iterable[ColorInsight]) -> list[dict]:
    """Return serializable rows for Plotly and Streamlit display."""
    return [
        {
            "Color": color.hex,
            "Share": round(color.percentage, 1),
            "Psychology": color.psychology,
        }
        for color in colors
    ]


def calculate_kpi_forecast(
    entropy_score: float,
    saliency_alignment: float,
    color_emotion: str,
) -> dict[str, float | str]:
    """Forecast paid-media KPIs from creative-quality signals.

    The model is intentionally lightweight and stateless. It starts from a mock
    benchmark and applies deterministic business rules that make the creative
    implications legible to a marketer.
    """
    base_cpc = 2.50
    base_conversion_rate = 0.015
    clutter = float(np.clip(entropy_score, 1, 100))
    saliency = float(np.clip(saliency_alignment, 0, 100))
    emotion = normalize_color_emotion(color_emotion)

    cpc = base_cpc
    conversion_rate = base_conversion_rate

    if clutter < 45:
        low_clutter_lift = np.clip((45 - clutter) / 35, 0, 1)
        cpc *= 1 - (0.15 * low_clutter_lift)
    elif clutter > 75:
        high_clutter_drag = np.clip((clutter - 75) / 25, 0, 1)
        cpc *= 1 + (0.10 * high_clutter_drag)
        conversion_rate *= 1 - (0.08 * high_clutter_drag)

    if saliency > 80:
        conversion_rate *= 1.20
    elif saliency < 45:
        conversion_rate *= 0.92

    if emotion == "Action/Urgency":
        conversion_rate *= 1.15
        cpc *= 1.05
    elif emotion == "Trust/Stability":
        conversion_rate *= 1.08
        cpc *= 0.98
    elif emotion == "Growth/Reassurance":
        conversion_rate *= 1.06
    elif emotion == "Premium/Aspiration":
        conversion_rate *= 1.04
        cpc *= 1.02
    elif emotion == "Neutral/Clarity":
        conversion_rate *= 1.02

    return {
        "predicted_cpc": f"${cpc:.2f}",
        "predicted_conversion_rate": f"{conversion_rate * 100:.2f}%",
        "cpc": round(float(cpc), 2),
        "conversion_rate": round(float(conversion_rate), 4),
        "color_emotion": emotion,
    }


def generate_persona_radar(entropy_score: float, color_profile: Iterable[ColorInsight] | str):
    """Build a Plotly radar chart mapping creative signals to buyer personas."""
    clutter = float(np.clip(entropy_score, 1, 100))
    low_clutter = 100 - clutter
    cool_weight, warm_weight, value_weight = _color_profile_weights(color_profile)

    millennial_balance = max(0, 100 - abs(clutter - 52) * 1.35)
    scores = [
        {
            "Persona": "B2B / Enterprise",
            "Score": _bounded_score(low_clutter * 0.62 + cool_weight * 0.38),
        },
        {
            "Persona": "Gen-Z Impulse",
            "Score": _bounded_score(clutter * 0.55 + warm_weight * 0.45),
        },
        {
            "Persona": "Millennial Value-Shopper",
            "Score": _bounded_score(millennial_balance * 0.44 + value_weight * 0.36 + warm_weight * 0.20),
        },
    ]

    fig = px.line_polar(
        scores,
        r="Score",
        theta="Persona",
        line_close=True,
        markers=True,
        range_r=[0, 100],
        title="Demographic Persona Matrix",
    )
    fig.update_traces(fill="toself", line_color="#FF7A00")
    fig.update_layout(
        height=430,
        margin=dict(l=30, r=30, t=70, b=30),
        polar=dict(radialaxis=dict(showticklabels=True, ticks="", range=[0, 100])),
        showlegend=False,
    )
    return fig


def micro_edit_prescriptions(results: dict) -> list[str]:
    """Generate 2-3 non-destructive creative edits from current image metrics."""
    clutter = int(results.get("clutter", 50))
    focus = float(results.get("focus_score", 50))
    emotion = float(results.get("emotion_score", 50))
    color_emotion = normalize_color_emotion(_dominant_emotion(results.get("colors", [])))

    prescriptions: list[str] = []
    if clutter > 75:
        prescriptions.append(
            f"Reduce background texture, small badges, or overlapping copy to lower clutter by about {min(clutter - 65, 25)} points."
        )
    elif clutter < 35:
        prescriptions.append(
            "Add one compact proof point or trust cue so the layout keeps clarity without feeling under-specified."
        )
    else:
        prescriptions.append(
            f"Keep visual density near the current {clutter}/100 level; adjust hierarchy through spacing before removing content."
        )

    if focus < 55:
        prescriptions.append(
            f"Move the CTA or product cue into the central 50% grid and increase contrast to gain roughly {round(60 - focus, 1)} saliency points."
        )
    else:
        prescriptions.append(
            "Preserve the current focal zone; apply edits around the edges so the main attention path stays intact."
        )

    if emotion < 30 and color_emotion != "Action/Urgency":
        prescriptions.append(
            "Introduce a small warm accent on the CTA to add action intent without changing the brand palette."
        )
    elif color_emotion == "Action/Urgency" and clutter > 70:
        prescriptions.append(
            "Keep the urgent CTA color, but mute one competing warm element so urgency reads as a single action cue."
        )
    else:
        prescriptions.append(
            "Retain the dominant color cue and test copy changes before making broader palette edits."
        )

    return prescriptions[:3]


def normalize_color_emotion(color_emotion: str) -> str:
    """Normalize psychology labels and segment labels into simulator options."""
    if color_emotion in COLOR_EMOTION_OPTIONS:
        return color_emotion
    return PSYCHOLOGY_TO_EMOTION.get(color_emotion, "Neutral/Clarity")


def _color_profile_weights(color_profile: Iterable[ColorInsight] | str) -> tuple[float, float, float]:
    if isinstance(color_profile, str):
        emotion = normalize_color_emotion(color_profile)
        warm = 85.0 if emotion == "Action/Urgency" else 35.0
        cool = 88.0 if emotion in {"Trust/Stability", "Growth/Reassurance"} else 35.0
        value = 82.0 if emotion in {"Growth/Reassurance", "Neutral/Clarity"} else 45.0
        if emotion == "Premium/Aspiration":
            value = 58.0
            cool = 62.0
        return cool, warm, value

    cool = 0.0
    warm = 0.0
    value = 0.0
    total = 0.0
    for color in color_profile:
        share = max(0.0, float(color.percentage))
        total += share
        emotion = normalize_color_emotion(color.psychology)
        if emotion == "Action/Urgency":
            warm += share
            value += share * 0.45
        elif emotion in {"Trust/Stability", "Growth/Reassurance"}:
            cool += share
            value += share * 0.85
        elif emotion == "Premium/Aspiration":
            cool += share * 0.55
            value += share * 0.55
        else:
            cool += share * 0.35
            value += share * 0.70

    if total <= 0:
        return 50.0, 50.0, 50.0
    return (
        float(np.clip(cool / total * 100, 0, 100)),
        float(np.clip(warm / total * 100, 0, 100)),
        float(np.clip(value / total * 100, 0, 100)),
    )


def _dominant_emotion(colors: Iterable[ColorInsight]) -> str:
    colors = list(colors)
    if not colors:
        return "Neutral/Clarity"
    return max(colors, key=lambda color: color.percentage).psychology


def _bounded_score(value: float) -> int:
    return int(round(float(np.clip(value, 0, 100))))


def _best_contrast_pair(colors: list[ColorInsight]) -> tuple[ColorInsight, ColorInsight, float]:
    if len(colors) < 2:
        fallback_dark = ColorInsight("#111111", 50.0, "Authority and Luxury")
        fallback_light = ColorInsight("#FFFFFF", 50.0, "Clarity and Simplicity")
        return fallback_dark, fallback_light, _contrast_ratio("#111111", "#FFFFFF")

    top_colors = colors[: min(5, len(colors))]
    best_pair = (top_colors[0], top_colors[1])
    best_ratio = 0.0
    for i, first in enumerate(top_colors):
        for second in top_colors[i + 1 :]:
            ratio = _contrast_ratio(first.hex, second.hex)
            share_weight = (first.percentage + second.percentage) / 200
            weighted_ratio = ratio * (0.7 + share_weight * 0.3)
            if weighted_ratio > best_ratio:
                best_ratio = weighted_ratio
                best_pair = (first, second)

    first, second = best_pair
    ratio = _contrast_ratio(first.hex, second.hex)
    if _relative_luminance(first.hex) <= _relative_luminance(second.hex):
        return first, second, ratio
    return second, first, ratio


def _contrast_ratio(first_hex: str, second_hex: str) -> float:
    first = _relative_luminance(first_hex)
    second = _relative_luminance(second_hex)
    lighter = max(first, second)
    darker = min(first, second)
    return float((lighter + 0.05) / (darker + 0.05))


def _relative_luminance(hex_code: str) -> float:
    rgb = np.array(_hex_to_rgb(hex_code), dtype=float) / 255.0
    linear = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])


def _wcag_status(ratio: float) -> str:
    if ratio >= 7:
        return "AAA Pass"
    if ratio >= 4.5:
        return "AA Pass"
    if ratio >= 3:
        return "Large Text Only"
    return "Fail Risk"


def _accessibility_recommendations(ratio: float, risk_pixels: float) -> list[str]:
    recommendations = []
    if ratio < 4.5:
        recommendations.append("Increase text/background contrast to at least 4.5:1 for normal CTA and body copy.")
    else:
        recommendations.append("Primary contrast pair is viable; preserve the dark/light separation during revisions.")

    if risk_pixels > 8:
        recommendations.append("Add a solid scrim or card behind text-dense regions flagged in the risk map.")
    elif risk_pixels > 3:
        recommendations.append("Check small legal copy and secondary badges; they may pass visually but fail at mobile sizes.")
    else:
        recommendations.append("Low-contrast risk regions are limited; prioritize message clarity over broad color changes.")

    if ratio < 3:
        recommendations.append("Avoid launching until headline and CTA contrast are corrected for accessibility and platform review.")
    else:
        recommendations.append("Run final QA at mobile crop sizes, where contrast failures become more likely.")
    return recommendations


def _accessibility_overlay(image_rgb: np.ndarray, risk_mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    red_layer = np.zeros_like(image_rgb)
    red_layer[:, :, 0] = 255
    red_layer[:, :, 1] = 42
    red_layer[:, :, 2] = 42
    alpha = np.where(risk_mask[..., None], 0.45, 0.0)
    overlay = (image_rgb * (1 - alpha) + red_layer * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(risk_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    return overlay


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


def _hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    normalized = hex_code.strip("#")
    return tuple(int(normalized[i : i + 2], 16) for i in (0, 2, 4))
