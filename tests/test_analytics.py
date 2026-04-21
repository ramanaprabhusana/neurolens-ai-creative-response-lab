import unittest

import numpy as np
from PIL import Image, ImageDraw

from analytics import (
    analyze_accessibility_contrast,
    calculate_kpi_forecast,
    calculate_entropy,
    clutter_score,
    compare_ads,
    extract_dominant_colors,
    generate_persona_radar,
    generate_attention_heatmap,
    generate_attention_heatmap_layers,
    micro_edit_prescriptions,
    score_ad,
)
from app import create_doctor_recommendation, create_preferred_symbol_asset, normalize_image_bytes, preferred_symbol_spec
from io import BytesIO
from telemetry import generate_attention_heatmap_layers_with_telemetry, score_ad_with_telemetry


def simple_ad() -> Image.Image:
    image = Image.new("RGB", (360, 240), "#1F5B73")
    draw = ImageDraw.Draw(image)
    draw.rectangle((36, 150, 165, 200), fill="#FF7A00")
    draw.text((52, 166), "BUY", fill="#FFFFFF")
    return image


def noisy_ad() -> Image.Image:
    rng = np.random.default_rng(9)
    return Image.fromarray(rng.integers(0, 255, (240, 360, 3), dtype=np.uint8), "RGB")


class AnalyticsTests(unittest.TestCase):
    def test_clutter_score_is_bounded(self):
        self.assertEqual(clutter_score(-1), 1)
        self.assertEqual(clutter_score(99), 100)

    def test_entropy_and_score_are_numeric(self):
        score = score_ad(simple_ad())
        self.assertGreaterEqual(score["entropy"], 0)
        self.assertGreaterEqual(score["clutter"], 1)
        self.assertLessEqual(score["clutter"], 100)
        self.assertIn("final_score", score)

    def test_noisy_image_scores_more_cluttered_than_simple_image(self):
        simple = score_ad(simple_ad())
        noisy = score_ad(noisy_ad())
        self.assertGreater(noisy["clutter"], simple["clutter"])

    def test_dominant_colors_return_three_or_fewer_insights(self):
        colors = extract_dominant_colors(simple_ad(), k=3)
        self.assertGreaterEqual(len(colors), 1)
        self.assertLessEqual(len(colors), 3)
        self.assertTrue(all(color.hex.startswith("#") for color in colors))

    def test_heatmap_shape_matches_resized_image(self):
        heatmap = generate_attention_heatmap(simple_ad())
        self.assertEqual(heatmap.shape, (240, 360, 3))

    def test_heatmap_layers_can_be_reused_without_recomputing_saliency(self):
        base, layer = generate_attention_heatmap_layers(simple_ad())
        self.assertEqual(base.shape, layer.shape)
        self.assertEqual(base.shape, (240, 360, 3))

    def test_compare_ads_returns_winner_and_scores(self):
        comparison = compare_ads(simple_ad(), noisy_ad())
        self.assertIn(comparison["winner"], {"Ad A", "Ad B"})
        self.assertIn("ad_a", comparison)
        self.assertIn("ad_b", comparison)

    def test_doctor_recommendation_returns_image(self):
        score = score_ad(simple_ad())
        recommendation = create_doctor_recommendation(simple_ad(), score)
        self.assertEqual(recommendation.size, (1100, 720))

    def test_preferred_symbol_is_generated_from_score(self):
        score = score_ad(simple_ad())
        spec = preferred_symbol_spec(score)
        symbol = create_preferred_symbol_asset(score)
        self.assertIn(spec["kind"], {"focus", "arrow", "shield", "spark", "tag"})
        self.assertEqual(symbol.size, (760, 500))

    def test_kpi_forecast_returns_formatted_business_metrics(self):
        forecast = calculate_kpi_forecast(30, 85, "Action/Urgency")
        self.assertTrue(forecast["predicted_cpc"].startswith("$"))
        self.assertTrue(forecast["predicted_conversion_rate"].endswith("%"))
        self.assertLess(forecast["cpc"], 2.50)
        self.assertGreater(forecast["conversion_rate"], 0.015)

    def test_persona_radar_returns_plotly_figure(self):
        colors = extract_dominant_colors(simple_ad(), k=3)
        fig = generate_persona_radar(35, colors)
        self.assertEqual(fig.data[0].type, "scatterpolar")
        self.assertIn("Demographic Persona Matrix", fig.layout.title.text)

    def test_micro_edit_prescriptions_are_specific(self):
        prescriptions = micro_edit_prescriptions(score_ad(noisy_ad()))
        self.assertGreaterEqual(len(prescriptions), 2)
        self.assertLessEqual(len(prescriptions), 3)
        self.assertTrue(all(isinstance(item, str) and item for item in prescriptions))

    def test_accessibility_contrast_returns_score_and_overlay(self):
        insight, overlay = analyze_accessibility_contrast(simple_ad())
        self.assertGreaterEqual(insight.score, 0)
        self.assertLessEqual(insight.score, 100)
        self.assertIn(insight.wcag_status, {"AAA Pass", "AA Pass", "Large Text Only", "Fail Risk"})
        self.assertEqual(overlay.shape, (240, 360, 3))

    def test_upload_normalization_handles_rgba(self):
        image = Image.new("RGBA", (80, 80), (255, 0, 0, 128))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        normalized = normalize_image_bytes(buffer.getvalue())
        self.assertEqual(normalized.mode, "RGB")

    def test_score_telemetry_includes_heavy_function_timers(self):
        score, timings, memory_mb = score_ad_with_telemetry(simple_ad())
        self.assertIn("final_score", score)
        self.assertIn("scikit_entropy_ms", timings)
        self.assertIn("kmeans_clustering_ms", timings)
        self.assertIn("opencv_edge_detection_ms", timings)
        self.assertGreater(memory_mb, 0)

    def test_heatmap_telemetry_includes_opencv_timers(self):
        base, layer, timings, memory_mb = generate_attention_heatmap_layers_with_telemetry(simple_ad())
        self.assertEqual(base.shape, layer.shape)
        self.assertIn("heatmap_opencv_edge_detection_ms", timings)
        self.assertIn("heatmap_layer_render_ms", timings)
        self.assertGreater(memory_mb, 0)


if __name__ == "__main__":
    unittest.main()
