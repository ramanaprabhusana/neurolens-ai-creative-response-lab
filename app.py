from __future__ import annotations

import json
import time
from datetime import datetime
from io import BytesIO

import cv2
import hashlib
import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from analytics import (
    COLOR_EMOTION_OPTIONS,
    analyze_accessibility_contrast,
    blend_heatmap_layers,
    calculate_kpi_forecast,
    colors_to_plotly_rows,
    generate_persona_radar,
    micro_edit_prescriptions,
    normalize_color_emotion,
    score_ad,
)
from telemetry import (
    begin_cache_probe,
    end_cache_probe,
    generate_attention_heatmap_layers_with_telemetry,
    mark_cache_miss,
    score_ad_with_telemetry,
)
from webrtc_callbacks import EmotionVideoProcessor


APP_NAME = "NeuroLens AI: Creative Response Lab"
SAMPLE_CREATIVES = {
    "Focused CTA": "focused",
    "Cluttered Retail": "cluttered",
    "Trust SaaS": "trust",
}
MAX_IMAGE_PIXELS = 24_000_000
MAX_ASPECT_RATIO = 8.0


st.set_page_config(page_title=APP_NAME, page_icon="N", layout="wide")


def main() -> None:
    inject_theme()
    render_sidebar()

    st.markdown(
        f"""
        <section class="nl-hero">
            <div class="nl-kicker">Computer Vision + Creative Analytics + Live Biometrics</div>
            <h1>{APP_NAME}</h1>
            <p>Audit advertising assets, predict A/B response, and compare the prediction with live webcam telemetry.</p>
            <div class="nl-badges">
                <span>No database</span>
                <span>Demo-ready samples</span>
                <span>Camera optional</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_showcase_story()

    tab_audit, tab_doctor, tab_ab, tab_simulator, tab_lab = st.tabs(
        ["Visual Asset Audit", "Creative Doctor", "A/B Predictor", "Campaign Simulator", "Neuromarketing Lab"]
    )

    with tab_audit:
        render_visual_asset_audit()

    with tab_doctor:
        render_creative_doctor()

    with tab_ab:
        render_ab_predictor()

    with tab_simulator:
        render_campaign_simulator()

    with tab_lab:
        render_neuromarketing_lab()


def render_visual_asset_audit() -> None:
    st.subheader("Visual Asset Audit")
    source_name, creative_bytes = creative_picker("Creative", "audit", default_kind="focused")
    if creative_bytes is None:
        st.info("Upload a JPG or PNG creative, or switch to a built-in sample.")
        return

    heatmap_alpha = st.slider(
        "Heatmap overlay opacity",
        min_value=0.25,
        max_value=0.75,
        value=0.50,
        step=0.05,
        key="audit_heatmap_alpha",
    )
    audit = analyze_creative(creative_bytes, source_name, heatmap_alpha=heatmap_alpha)
    if audit is None:
        return
    image, results, heatmap = audit
    accessibility = analyze_accessibility_safely(creative_bytes, source_name)

    left, right = st.columns([1.05, 0.95], vertical_alignment="top")
    with left:
        st.image(image, caption=f"{source_name} creative", use_container_width=True)
        st.image(heatmap, caption="Simulated attention heatmap", use_container_width=True)
        render_heatmap_download(heatmap, source_name, key="audit_heatmap")
        if accessibility is not None:
            _, accessibility_overlay = accessibility
            st.image(
                accessibility_overlay,
                caption="Accessibility contrast risk overlay",
                use_container_width=True,
            )

    with right:
        render_audit_metrics(results)
        render_insight_summary(results)
        render_recommendations(results)
        render_micro_edit_prescriptions(results)
        render_business_forecast(results)
        if accessibility is not None:
            render_accessibility_panel(accessibility[0])
        render_audit_download(
            source_name,
            results,
            key="audit_report",
            accessibility=accessibility[0] if accessibility is not None else None,
        )
        render_color_chart(results["colors"])
        render_persona_matrix(results)


def render_creative_doctor() -> None:
    st.subheader("Creative Doctor")
    source_name, creative_bytes = creative_picker("Creative Doctor input", "doctor", default_kind="cluttered")
    if creative_bytes is None:
        st.info("Select a sample or upload a creative to generate a cleaner recommended layout.")
        return

    image = load_image_safely(creative_bytes, source_name)
    if image is None:
        return

    with st.spinner("Diagnosing creative and generating recommended layout..."):
        original_score = get_cached_score(creative_bytes, "Doctor Original Score")
        doctor_bytes = cached_doctor_creative_bytes(creative_bytes)
        doctor_image = cached_image_from_bytes(doctor_bytes)
        doctor_score = get_cached_score(doctor_bytes, "Doctor Recommendation Score")
        comparison = compare_score_dicts(original_score, doctor_score)

    render_scorecard("Current creative", original_score)
    render_doctor_delta(original_score, doctor_score)

    before_col, after_col = st.columns(2, vertical_alignment="top")
    with before_col:
        st.image(image, caption=f"Before: {source_name}", use_container_width=True)
        render_recommendations(original_score)

    with after_col:
        st.image(doctor_image, caption="Creative Doctor recommendation", use_container_width=True)
        render_recommendations(doctor_score)
        st.download_button(
            "Download recommended PNG",
            data=doctor_bytes,
            file_name=f"neurolens_doctor_{safe_slug(source_name)}.png",
            mime="image/png",
            key="doctor_png",
            use_container_width=True,
        )

    st.caption("Doctor mode creates a stateless mock redesign from the detected palette and audit signals.")
    render_doctor_download(source_name, original_score, doctor_score, comparison)


def render_ab_predictor() -> None:
    st.subheader("A/B Predictor")
    col_a, col_b = st.columns(2)
    with col_a:
        name_a, bytes_a = creative_picker("Ad A", "ab_a", default_kind="focused")
    with col_b:
        name_b, bytes_b = creative_picker("Ad B", "ab_b", default_kind="cluttered")

    if bytes_a is None or bytes_b is None:
        st.info("Select or upload both ads to run the simultaneous entropy and saliency comparison.")
        return

    image_a = load_image_safely(bytes_a, name_a)
    image_b = load_image_safely(bytes_b, name_b)
    if image_a is None or image_b is None:
        return

    heatmap_alpha = st.slider(
        "Heatmap overlay opacity",
        min_value=0.25,
        max_value=0.75,
        value=0.50,
        step=0.05,
        key="ab_heatmap_alpha",
    )
    with st.spinner("Running A/B saliency and entropy model..."):
        score_a = get_cached_score(bytes_a, "Ad A Visual Analysis")
        score_b = get_cached_score(bytes_b, "Ad B Visual Analysis")
        comparison = compare_score_dicts(score_a, score_b)
        heatmap_a = get_cached_heatmap(bytes_a, heatmap_alpha, "Ad A Saliency Heatmap")
        heatmap_b = get_cached_heatmap(bytes_b, heatmap_alpha, "Ad B Saliency Heatmap")

    st.success(
        f"Predicted Winner: {comparison['winner']} - "
        f"{confidence_label(comparison['margin'])} confidence - "
        f"Margin: {comparison['margin']} pts"
    )
    render_winner_rationale(comparison)

    score_cols = st.columns(2)
    render_ad_score(score_cols[0], "Ad A", name_a, image_a, comparison["ad_a"], heatmap_a)
    render_ad_score(score_cols[1], "Ad B", name_b, image_b, comparison["ad_b"], heatmap_b)
    render_ab_download(name_a, name_b, comparison)
    render_ab_chart(comparison)


def render_campaign_simulator() -> None:
    st.subheader("Campaign Simulator")
    st.caption("Run a stateless what-if model before you touch the creative file.")

    input_col, output_col = st.columns([0.9, 1.1], vertical_alignment="top")
    with input_col:
        simulated_clutter = st.slider(
            "Simulated Clutter (Entropy)",
            min_value=10,
            max_value=100,
            value=48,
            step=1,
            key="simulated_clutter",
        )
        saliency_alignment = st.slider(
            "Saliency Grid Alignment",
            min_value=0,
            max_value=100,
            value=72,
            step=1,
            format="%d%%",
            key="simulated_saliency",
        )
        color_emotion = st.selectbox(
            "Dominant Color Psychology",
            COLOR_EMOTION_OPTIONS,
            index=COLOR_EMOTION_OPTIONS.index("Action/Urgency"),
            key="simulated_color_emotion",
        )

    forecast = calculate_kpi_forecast(simulated_clutter, saliency_alignment, color_emotion)
    simulator_results = {
        "clutter": simulated_clutter,
        "focus_score": saliency_alignment,
        "emotion_score": simulator_emotion_score(color_emotion),
        "colors": [],
    }

    with output_col:
        cols = st.columns(3)
        cols[0].metric("Predicted CPC", forecast["predicted_cpc"])
        cols[1].metric("Predicted CVR", forecast["predicted_conversion_rate"])
        cols[2].metric("Color Segment", forecast["color_emotion"])
        st.plotly_chart(
            generate_persona_radar(simulated_clutter, color_emotion),
            use_container_width=True,
        )
        render_micro_edit_prescriptions(simulator_results, heading="What-if edit plan")


def render_neuromarketing_lab() -> None:
    st.subheader("Neuromarketing Lab")
    source_name, creative_bytes = creative_picker("Stimulus", "lab", default_kind="trust")
    if creative_bytes is None:
        st.info("Upload a lab stimulus, or switch to a built-in sample.")
        return

    audit = analyze_creative(creative_bytes, source_name, heatmap_alpha=0.50)
    if audit is None:
        return
    stimulus, predicted, _ = audit

    ad_col, webcam_col = st.columns([1, 1], vertical_alignment="top")
    with ad_col:
        st.image(stimulus, caption=f"{source_name} stimulus", use_container_width=True)
        st.caption("Predicted response model")
        prediction_cols = st.columns(3)
        prediction_cols[0].metric("Clutter", f"{predicted['clutter']}/100")
        prediction_cols[1].metric("Emotion Fit", f"{predicted['emotion_score']}/100")
        prediction_cols[2].metric("Saliency", f"{predicted['focus_score']}/100")
        render_recommendations(predicted)
        render_audit_download(source_name, predicted, key="lab_prediction_report")

    with webcam_col:
        st.caption("Live biometric telemetry")
        ctx = webrtc_streamer(
            key="neuromarketing-lab",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

        metric_slots = st.container()
        status = st.empty()

        if ctx.state.playing:
            processor = ctx.video_processor
            metrics = processor.get_metrics() if processor else {}
            render_live_metrics(metric_slots, metrics)
            if metrics.get("face_detected"):
                status.success("Face signal detected. Latest-frame telemetry is active.")
            else:
                status.info("Camera is active. Center your face in frame to begin telemetry.")
            time.sleep(0.35)
            st.rerun()
        else:
            render_live_metrics(metric_slots, {})
            status.info("Start the camera to compare predicted response with live facial telemetry. If access is denied, the rest of the suite stays usable.")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### NeuroLens AI")
        st.metric("Runtime", "Stateless")
        st.metric("Storage", "No database")
        st.divider()
        st.markdown("**Modules**")
        st.caption("UI: app.py")
        st.caption("Analytics: analytics.py")
        st.caption("Video: webrtc_callbacks.py")
        st.divider()
        st.caption(
            "Facial telemetry is heuristic and frame-by-frame. It is a demo signal, not a clinical emotion classifier."
        )
        render_developer_telemetry()
        with st.expander("Model methodology"):
            st.markdown(
                """
                - Clutter: Shannon entropy normalized to a 1-100 scale.
                - Saliency: OpenCV edges plus local contrast, blurred into a simulated heatmap.
                - Color psychology: K-Means dominant colors mapped to nearest psychology labels.
                - Business forecast: deterministic mock CPC/CVR rules, not ad-platform estimates.
                - Persona matrix: clutter and color-temperature mapping to audience-fit scores.
                - Live telemetry: latest-frame face-region heuristics only.
                """
            )


def render_showcase_story() -> None:
    st.markdown(
        """
        <section class="nl-story-grid">
            <div class="nl-before-after">
                <h3>The old way</h3>
                <p>Guessing which ad is too busy, manually pulling colors, and separating creative prediction from user response.</p>
            </div>
            <div class="nl-before-after nl-after">
                <h3>With NeuroLens</h3>
                <p>Entropy, saliency, color psychology, A/B scoring, and live biometric telemetry sit in one local-first workflow.</p>
            </div>
            <div class="nl-steps">
                <div><span>1</span><strong>Audit</strong><small>Score visual load and attention.</small></div>
                <div><span>2</span><strong>Compare</strong><small>Predict the stronger creative.</small></div>
                <div><span>3</span><strong>Measure</strong><small>Watch live response signals.</small></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_developer_telemetry() -> None:
    telemetry = st.session_state.get("developer_telemetry", {})
    with st.sidebar.expander("⚙️ Developer Telemetry (Dev Mode)", expanded=False):
        if not telemetry:
            st.caption("Run an audit to populate computer vision latency and cache metrics.")
            return

        for label, event in telemetry.items():
            st.markdown(f"**{label}**")
            cols = st.columns(3)
            cols[0].metric("Cache", event["cache_status"])
            cols[1].metric("Wall", f"{event['wall_ms']:.1f} ms")
            cols[2].metric("Memory", f"{event['memory_mb']:.2f} MB")

            timing_rows = event.get("timings_ms", {})
            if timing_rows:
                for metric_name, value in timing_rows.items():
                    st.metric(format_telemetry_label(metric_name), f"{value:.2f} ms")
            st.divider()

        st.caption("Cache Hit means Streamlit returned cached data and the cached function body did not execute.")


def record_developer_telemetry(
    label: str,
    cache_status: str,
    wall_ms: float,
    timings_ms: dict[str, float],
    memory_mb: float,
) -> None:
    telemetry = dict(st.session_state.get("developer_telemetry", {}))
    telemetry[label] = {
        "cache_status": cache_status,
        "wall_ms": round(float(wall_ms), 2),
        "timings_ms": dict(timings_ms),
        "memory_mb": round(float(memory_mb), 2),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state["developer_telemetry"] = telemetry

    toast_key = f"telemetry_toast_{label}"
    previous_status = st.session_state.get(toast_key)
    if previous_status != cache_status:
        st.toast(f"{label}: Cache {cache_status}", icon="⚙️")
        st.session_state[toast_key] = cache_status


def format_telemetry_label(metric_name: str) -> str:
    return metric_name.replace("_ms", "").replace("_", " ").title()


def creative_picker(label: str, key: str, default_kind: str) -> tuple[str, bytes | None]:
    mode = st.radio(
        f"{label} source",
        ["Sample", "Upload"],
        horizontal=True,
        key=f"{key}_mode",
    )
    if mode == "Sample":
        labels = list(SAMPLE_CREATIVES)
        default_label = next(
            (sample_label for sample_label, kind in SAMPLE_CREATIVES.items() if kind == default_kind),
            labels[0],
        )
        selected = st.selectbox(
            f"{label} sample",
            labels,
            index=labels.index(default_label),
            key=f"{key}_sample",
        )
        return selected, sample_creative_bytes(SAMPLE_CREATIVES[selected])

    uploaded = st.file_uploader(
        f"Upload {label} JPG or PNG",
        type=["jpg", "jpeg", "png"],
        key=f"{key}_upload",
    )
    if uploaded is None:
        return label, None
    return uploaded.name, uploaded.getvalue()


def analyze_creative(
    file_bytes: bytes,
    source_name: str,
    heatmap_alpha: float,
) -> tuple[Image.Image, dict, np.ndarray] | None:
    image = load_image_safely(file_bytes, source_name)
    if image is None:
        return None
    try:
        with st.spinner("Running computer vision audit..."):
            return image, get_cached_score(file_bytes, "Visual Analysis"), get_cached_heatmap(
                file_bytes,
                heatmap_alpha,
                "Saliency Heatmap",
            )
    except (cv2.error, ValueError, RuntimeError) as exc:
        st.error("The image loaded, but the computer vision audit could not process it safely.")
        st.caption(str(exc))
        return None


def analyze_accessibility_safely(file_bytes: bytes, source_name: str):
    try:
        with st.spinner("Scanning contrast and accessibility risk..."):
            return cached_accessibility_from_bytes(file_bytes)
    except (cv2.error, ValueError, RuntimeError) as exc:
        st.warning(f"{source_name} could not be scanned for contrast risk.")
        st.caption(str(exc))
        return None


def load_image_safely(file_bytes: bytes, source_name: str) -> Image.Image | None:
    try:
        return cached_image_from_bytes(file_bytes)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        st.error(f"{source_name} could not be read as a valid image. Try a different JPG or PNG.")
        st.caption(str(exc))
        return None


@st.cache_data(show_spinner=False)
def cached_image_from_bytes(file_bytes: bytes) -> Image.Image:
    return normalize_image_bytes(file_bytes)


def normalize_image_bytes(file_bytes: bytes) -> Image.Image:
    """Load uploads defensively and normalize RGBA/CMYK/P-mode images to RGB."""
    if not file_bytes:
        raise ValueError("Uploaded image is empty.")

    with Image.open(BytesIO(file_bytes)) as image:
        image.load()
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions.")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image is too large for interactive analysis. Resize below {MAX_IMAGE_PIXELS:,} pixels."
            )
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > MAX_ASPECT_RATIO:
            raise ValueError(
                f"Extreme aspect ratio detected ({aspect_ratio:.1f}:1). Crop the creative closer to its ad placement."
            )

        if image.mode == "CMYK":
            return image.convert("RGB")
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            return background.convert("RGB")
        return image.convert("RGB")


@st.cache_data(show_spinner=False)
def cached_score_payload_from_bytes(file_bytes: bytes, cache_label: str) -> dict:
    mark_cache_miss(cache_label)
    score, timings_ms, memory_mb = score_ad_with_telemetry(cached_image_from_bytes(file_bytes))
    return {"score": score, "timings_ms": timings_ms, "memory_mb": memory_mb}


@st.cache_data(show_spinner=False)
def cached_score_from_bytes(file_bytes: bytes) -> dict:
    return cached_score_payload_from_bytes(file_bytes, cache_probe_label("score", file_bytes))["score"]


@st.cache_data(show_spinner=False)
def cached_attention_payload_from_bytes(file_bytes: bytes, cache_label: str) -> dict:
    mark_cache_miss(cache_label)
    image_rgb, color_heatmap, timings_ms, memory_mb = generate_attention_heatmap_layers_with_telemetry(
        cached_image_from_bytes(file_bytes)
    )
    return {
        "image_rgb": image_rgb,
        "color_heatmap": color_heatmap,
        "timings_ms": timings_ms,
        "memory_mb": memory_mb,
    }


@st.cache_data(show_spinner=False)
def cached_attention_layers_from_bytes(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    payload = cached_attention_payload_from_bytes(file_bytes, cache_probe_label("heatmap", file_bytes))
    return payload["image_rgb"], payload["color_heatmap"]


@st.cache_data(show_spinner=False)
def cached_heatmap_from_bytes(file_bytes: bytes, alpha: float) -> np.ndarray:
    image_rgb, color_heatmap = cached_attention_layers_from_bytes(file_bytes)
    return blend_heatmap_layers(image_rgb, color_heatmap, alpha=alpha)


def get_cached_score(file_bytes: bytes, label: str) -> dict:
    cache_label = cache_probe_label("score", file_bytes)
    previous_count = begin_cache_probe(cache_label)
    started = time.perf_counter()
    payload = cached_score_payload_from_bytes(file_bytes, cache_label)
    wall_ms = (time.perf_counter() - started) * 1000
    cache_status = end_cache_probe(cache_label, previous_count)
    record_developer_telemetry(
        label=label,
        cache_status=cache_status,
        wall_ms=wall_ms,
        timings_ms=payload["timings_ms"],
        memory_mb=payload["memory_mb"],
    )
    return payload["score"]


def get_cached_heatmap(file_bytes: bytes, alpha: float, label: str) -> np.ndarray:
    cache_label = cache_probe_label("heatmap", file_bytes)
    previous_count = begin_cache_probe(cache_label)
    started = time.perf_counter()
    payload = cached_attention_payload_from_bytes(file_bytes, cache_label)
    heatmap = blend_heatmap_layers(payload["image_rgb"], payload["color_heatmap"], alpha=alpha)
    wall_ms = (time.perf_counter() - started) * 1000
    cache_status = end_cache_probe(cache_label, previous_count)
    timings = dict(payload["timings_ms"])
    timings["heatmap_alpha_blend_ms"] = round(wall_ms if cache_status == "Hit" else max(0.0, wall_ms - sum(timings.values())), 2)
    record_developer_telemetry(
        label=label,
        cache_status=cache_status,
        wall_ms=wall_ms,
        timings_ms=timings,
        memory_mb=payload["memory_mb"],
    )
    return heatmap


def cache_probe_label(namespace: str, file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"{namespace}:{digest}"


@st.cache_data(show_spinner=False)
def cached_accessibility_from_bytes(file_bytes: bytes):
    return analyze_accessibility_contrast(cached_image_from_bytes(file_bytes))


@st.cache_data(show_spinner=False)
def cached_compare_ads(bytes_a: bytes, bytes_b: bytes) -> dict:
    return compare_score_dicts(cached_score_from_bytes(bytes_a), cached_score_from_bytes(bytes_b))


@st.cache_data(show_spinner=False)
def cached_doctor_creative_bytes(file_bytes: bytes) -> bytes:
    image = cached_image_from_bytes(file_bytes)
    score = cached_score_from_bytes(file_bytes)
    doctor_image = create_doctor_recommendation(image, score)
    buffer = BytesIO()
    doctor_image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def sample_creative_bytes(kind: str) -> bytes:
    image = create_sample_ad(kind)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_audit_metrics(results: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("Visual Clutter", f"{results['clutter']}/100", help="Derived from Shannon entropy.")
    cols[1].metric("Entropy", f"{results['entropy']:.2f}")
    cols[2].metric("Central Saliency", f"{results['focus_score']}/100")

    if results["clutter"] > 75:
        st.warning("High cognitive load detected. Simplify competing detail or strengthen visual hierarchy.")
    else:
        st.success("Cognitive load is within the recommended operating range.")


def render_scorecard(label: str, score: dict) -> None:
    grade, launch = creative_grade(score)
    st.markdown(
        f"""
        <div class="nl-scorecard">
            <span>{label}</span>
            <strong>Creative Readiness: {grade}</strong>
            <small>{launch}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_doctor_delta(original_score: dict, doctor_score: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Clutter Change", f"{doctor_score['clutter'] - original_score['clutter']:+d}")
    cols[1].metric("Saliency Change", f"{doctor_score['focus_score'] - original_score['focus_score']:+.1f}")
    cols[2].metric("Emotion Change", f"{doctor_score['emotion_score'] - original_score['emotion_score']:+.1f}")
    cols[3].metric("Final Score Change", f"{doctor_score['final_score'] - original_score['final_score']:+.1f}")


def render_recommendations(results: dict) -> None:
    st.markdown("**Recommended next actions**")
    st.markdown("\n".join(f"- {recommendation}" for recommendation in audit_recommendations(results)))


def render_micro_edit_prescriptions(results: dict, heading: str = "Micro-edit prescriptions") -> None:
    st.markdown(f"**{heading}**")
    st.markdown("\n".join(f"- {instruction}" for instruction in micro_edit_prescriptions(results)))


def render_business_forecast(results: dict) -> None:
    color_emotion = dominant_color_emotion(results)
    forecast = calculate_kpi_forecast(results["clutter"], results["focus_score"], color_emotion)
    st.markdown("**Predictive ROI & CPC forecast**")
    cols = st.columns(2)
    cols[0].metric("Predicted CPC", forecast["predicted_cpc"])
    cols[1].metric("Predicted CVR", forecast["predicted_conversion_rate"])
    st.caption(f"Baseline adjusted from $2.50 CPC and 1.5% CVR using {forecast['color_emotion']} cues.")


def render_accessibility_panel(accessibility) -> None:
    st.markdown("**Accessibility & platform risk scanner**")
    cols = st.columns(4)
    cols[0].metric("Access Score", f"{accessibility.score}/100")
    cols[1].metric("WCAG", accessibility.wcag_status)
    cols[2].metric("Contrast", f"{accessibility.contrast_ratio}:1")
    cols[3].metric("Risk Pixels", f"{accessibility.risk_pixels:.1f}%")
    st.markdown(
        f"""
        <div class="nl-contrast-pair">
            <span style="background:{accessibility.foreground_hex};"></span>
            <strong>{accessibility.foreground_hex}</strong>
            <span style="background:{accessibility.background_hex};"></span>
            <strong>{accessibility.background_hex}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Contrast remediation plan", expanded=accessibility.contrast_ratio < 4.5):
        st.markdown("\n".join(f"- {item}" for item in accessibility.recommendations))


def render_persona_matrix(results: dict) -> None:
    st.plotly_chart(
        generate_persona_radar(results["clutter"], results.get("colors", [])),
        use_container_width=True,
    )


def render_insight_summary(results: dict) -> None:
    dominant = results["colors"][0] if results["colors"] else None
    clutter_label = "Needs simplification" if results["clutter"] > 75 else "Readable"
    focus_label = "Strong central focus" if results["focus_score"] >= 60 else "Diffuse attention"
    emotion_label = dominant.psychology if dominant else "Neutral"

    st.markdown(
        f"""
        <div class="nl-insight-row">
            <div class="nl-insight"><span>Clarity</span><strong>{clutter_label}</strong></div>
            <div class="nl-insight"><span>Primary Cue</span><strong>{emotion_label}</strong></div>
            <div class="nl-insight"><span>Attention</span><strong>{focus_label}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_winner_rationale(comparison: dict) -> None:
    winner_key = "ad_a" if comparison["winner"] == "Ad A" else "ad_b"
    loser_key = "ad_b" if winner_key == "ad_a" else "ad_a"
    winner = comparison[winner_key]
    loser = comparison[loser_key]

    reasons = []
    if winner["clutter"] < loser["clutter"]:
        reasons.append("lower visual clutter")
    if winner["emotion_score"] >= loser["emotion_score"]:
        reasons.append("stronger target emotion fit")
    if winner["focus_score"] >= loser["focus_score"]:
        reasons.append("stronger central saliency")
    if not reasons:
        reasons.append("a better blended model score")

    st.info(f"{comparison['winner']} wins on {', '.join(reasons)}.")


def compare_score_dicts(score_a: dict, score_b: dict) -> dict:
    winner = "Ad A" if score_a["final_score"] >= score_b["final_score"] else "Ad B"
    margin = abs(score_a["final_score"] - score_b["final_score"])
    return {"ad_a": score_a, "ad_b": score_b, "winner": winner, "margin": round(margin, 1)}


def confidence_label(margin: float) -> str:
    if margin >= 20:
        return "High"
    if margin >= 10:
        return "Medium"
    return "Exploratory"


def creative_grade(score: dict) -> tuple[str, str]:
    final_score = float(score.get("final_score", 0))
    clutter = int(score["clutter"])
    if clutter > 85:
        return "C-", "Do not launch yet. Simplify visual hierarchy first."
    if final_score >= 70:
        return "A", "Ready for a high-confidence test."
    if final_score >= 55:
        return "B", "Ready to test with minor creative refinements."
    if final_score >= 40:
        return "C", "Needs a focused creative revision before launch."
    return "D", "Rework the message, focal point, and CTA before testing."


def dominant_color_emotion(results: dict) -> str:
    colors = results.get("colors", [])
    if not colors:
        return "Neutral/Clarity"
    dominant = max(colors, key=lambda color: color.percentage)
    return normalize_color_emotion(dominant.psychology)


def simulator_emotion_score(color_emotion: str) -> float:
    if color_emotion == "Action/Urgency":
        return 72.0
    if color_emotion in {"Trust/Stability", "Growth/Reassurance"}:
        return 58.0
    if color_emotion == "Premium/Aspiration":
        return 52.0
    return 38.0


def render_audit_download(source_name: str, results: dict, key: str, accessibility=None) -> None:
    payload = {
        "app": APP_NAME,
        "report_type": "creative_audit",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "creative": source_name,
        "scores": serializable_score(results),
        "recommendations": audit_recommendations(results),
        "micro_edit_prescriptions": micro_edit_prescriptions(results),
        "business_forecast": calculate_kpi_forecast(
            results["clutter"],
            results["focus_score"],
            dominant_color_emotion(results),
        ),
        "model_notes": [
            "Visual clutter is derived from Shannon entropy over grayscale luminance.",
            "Attention heatmap is simulated from OpenCV edge and contrast signals.",
            "Color psychology uses nearest-color matching against a predefined dictionary.",
            "Business forecast uses mock CPC/CVR rules for scenario planning.",
        ],
    }
    if accessibility is not None:
        payload["accessibility_risk"] = {
            "score": accessibility.score,
            "wcag_status": accessibility.wcag_status,
            "contrast_ratio": accessibility.contrast_ratio,
            "risk_pixels": accessibility.risk_pixels,
            "foreground_hex": accessibility.foreground_hex,
            "background_hex": accessibility.background_hex,
            "recommendations": list(accessibility.recommendations),
        }
    render_json_download("Download audit report", f"neurolens_audit_{safe_slug(source_name)}.json", payload, key)


def render_ab_download(name_a: str, name_b: str, comparison: dict) -> None:
    payload = {
        "app": APP_NAME,
        "report_type": "ab_prediction",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ad_a": {"name": name_a, "scores": serializable_score(comparison["ad_a"])},
        "ad_b": {"name": name_b, "scores": serializable_score(comparison["ad_b"])},
        "predicted_winner": comparison["winner"],
        "confidence_margin": comparison["margin"],
        "confidence_label": confidence_label(comparison["margin"]),
        "winner_rationale": winner_reason_list(comparison),
        "model_formula": "final = low_clutter*0.45 + emotion_fit*0.35 + central_saliency*0.20",
    }
    render_json_download("Download A/B report", "neurolens_ab_prediction.json", payload, "ab_report")


def render_doctor_download(source_name: str, original_score: dict, doctor_score: dict, comparison: dict) -> None:
    original_grade, original_launch = creative_grade(original_score)
    doctor_grade, doctor_launch = creative_grade(doctor_score)
    payload = {
        "app": APP_NAME,
        "report_type": "creative_doctor",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "creative": source_name,
        "original": {
            "grade": original_grade,
            "launch_recommendation": original_launch,
            "scores": serializable_score(original_score),
            "recommendations": audit_recommendations(original_score),
        },
        "doctor_recommendation": {
            "grade": doctor_grade,
            "launch_recommendation": doctor_launch,
            "scores": serializable_score(doctor_score),
            "recommendations": audit_recommendations(doctor_score),
        },
        "doctor_vs_original": {
            "predicted_winner": comparison["winner"],
            "confidence_margin": comparison["margin"],
            "confidence_label": confidence_label(comparison["margin"]),
        },
    }
    render_json_download("Download Creative Doctor report", "neurolens_creative_doctor.json", payload, "doctor_report")


def render_json_download(label: str, filename: str, payload: dict, key: str) -> None:
    st.download_button(
        label,
        data=json.dumps(payload, indent=2),
        file_name=filename,
        mime="application/json",
        key=key,
        use_container_width=True,
    )


def audit_recommendations(results: dict) -> list[str]:
    recommendations = []
    if results["clutter"] > 75:
        recommendations.append("Reduce competing visual detail or isolate the primary CTA with more whitespace.")
    elif results["clutter"] < 30:
        recommendations.append("Creative is very clean; consider adding one stronger proof point if conversion context is missing.")
    else:
        recommendations.append("Maintain the current visual density while keeping the main CTA visually isolated.")

    if results["focus_score"] < 55:
        recommendations.append("Move the product, face, or CTA closer to the central attention zone.")
    else:
        recommendations.append("Central attention is healthy; preserve the current focal hierarchy.")

    if results["emotion_score"] < 30:
        recommendations.append("Consider warmer action colors or more emotionally explicit messaging for acquisition campaigns.")
    else:
        recommendations.append("Color psychology is directionally strong for action or trust cues.")

    return recommendations


def winner_reason_list(comparison: dict) -> list[str]:
    winner_key = "ad_a" if comparison["winner"] == "Ad A" else "ad_b"
    loser_key = "ad_b" if winner_key == "ad_a" else "ad_a"
    winner = comparison[winner_key]
    loser = comparison[loser_key]
    reasons = []
    if winner["clutter"] < loser["clutter"]:
        reasons.append("lower visual clutter")
    if winner["emotion_score"] >= loser["emotion_score"]:
        reasons.append("stronger target emotion fit")
    if winner["focus_score"] >= loser["focus_score"]:
        reasons.append("stronger central saliency")
    return reasons or ["better blended model score"]


def serializable_score(score: dict) -> dict:
    return {
        "entropy": round(float(score["entropy"]), 3),
        "visual_clutter": int(score["clutter"]),
        "emotion_score": float(score["emotion_score"]),
        "central_saliency": float(score["focus_score"]),
        "final_score": float(score.get("final_score", 0)),
        "dominant_colors": [
            {
                "hex": color.hex,
                "share": round(float(color.percentage), 1),
                "psychology": color.psychology,
            }
            for color in score.get("colors", [])
        ],
    }


def safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return slug or "creative"


def render_color_chart(colors) -> None:
    rows = colors_to_plotly_rows(colors)
    fig = px.bar(
        rows,
        x="Color",
        y="Share",
        color="Color",
        color_discrete_map={row["Color"]: row["Color"] for row in rows},
        text="Psychology",
        title="Top Dominant Colors and Psychological Associations",
        range_y=[0, 100],
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=70, b=20))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    swatches = st.columns(max(1, len(colors)))
    for column, color in zip(swatches, colors):
        with column:
            st.markdown(
                f"""
                <div class="nl-swatch" style="background:{color.hex};"></div>
                <strong>{color.hex}</strong><br>
                <span>{color.percentage:.1f}%</span><br>
                <small>{color.psychology}</small>
                """,
                unsafe_allow_html=True,
            )


def render_ad_score(column, label: str, source_name: str, image: Image.Image, score: dict, heatmap: np.ndarray) -> None:
    with column:
        st.image(image, caption=f"{label}: {source_name}", use_container_width=True)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Final", score["final_score"])
        metric_cols[1].metric("Clutter", score["clutter"])
        metric_cols[2].metric("Emotion", score["emotion_score"])
        st.image(heatmap, caption=f"{label} heatmap", use_container_width=True)
        render_heatmap_download(heatmap, f"{label}_{source_name}", key=f"{label.lower().replace(' ', '_')}_heatmap")


def render_heatmap_download(heatmap: np.ndarray, source_name: str, key: str) -> None:
    st.download_button(
        "Download heatmap PNG",
        data=image_array_to_png_bytes(heatmap),
        file_name=f"neurolens_heatmap_{safe_slug(source_name)}.png",
        mime="image/png",
        key=key,
        use_container_width=True,
    )


def image_array_to_png_bytes(image_array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image_array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def create_doctor_recommendation(image: Image.Image, score: dict) -> Image.Image:
    width, height = 1100, 720
    dominant = score["colors"][0].hex if score.get("colors") else "#1F5B73"
    accent = best_accent_hex(score)
    background = soften_hex(dominant, amount=0.88)
    text_color = "#102F3B"

    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((70, 70, width - 70, height - 70), radius=26, fill="#FFFFFF", outline="#D9E4EA", width=4)

    headline = "Clarity that converts."
    subhead = doctor_subhead(score)
    cta = "TEST THIS VERSION"

    draw.text((125, 135), headline, fill=text_color, font=_font(60))
    draw.text((130, 225), subhead, fill="#42616D", font=_font(34))

    draw.rounded_rectangle((130, 500, 500, 585), radius=20, fill=accent)
    draw.text((164, 526), cta, fill="#FFFFFF", font=_font(30))

    draw.rounded_rectangle((385, 95, 1030, 625), radius=28, fill=accent, outline=accent, width=5)
    draw.ellipse((705, 210, 880, 385), fill="#FFFFFF", outline=dominant, width=6)
    draw.arc((740, 262, 845, 348), 200, 340, fill=accent, width=7)
    draw.ellipse((762, 275, 785, 298), fill=text_color)
    draw.ellipse((812, 275, 835, 298), fill=text_color)
    draw.text((660, 455), "Primary focal zone", fill="#FFFFFF", font=_font(30))
    return canvas


def doctor_subhead(score: dict) -> str:
    if score["clutter"] > 75:
        return "Reduced clutter, stronger CTA isolation, cleaner focal hierarchy."
    if score["focus_score"] < 55:
        return "Recentered the CTA and product cue for clearer attention flow."
    return "Preserved the strongest cue while simplifying the conversion path."


def best_accent_hex(score: dict) -> str:
    preferred = ["High Urgency", "Energy and Action", "Optimism and Attention"]
    for psychology in preferred:
        for color in score.get("colors", []):
            if color.psychology == psychology:
                return color.hex
    for color in score.get("colors", []):
        if color.psychology in {"Growth and Reassurance", "Trust and Stability", "Premium and Imagination"}:
            return color.hex
    return "#FF7A00"


def soften_hex(hex_code: str, amount: float) -> str:
    rgb = np.array(hex_to_rgb(hex_code), dtype=float)
    softened = rgb + (255 - rgb) * amount
    return rgb_to_hex(np.clip(softened, 0, 255).astype(int))


def rgb_to_hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def hex_to_rgb(hex_code: str) -> tuple[int, int, int]:
    normalized = hex_code.strip("#")
    return tuple(int(normalized[i : i + 2], 16) for i in (0, 2, 4))


def render_ab_chart(comparison: dict) -> None:
    fig = px.bar(
        [
            {"Ad": "Ad A", "Metric": "Final Score", "Score": comparison["ad_a"]["final_score"]},
            {"Ad": "Ad A", "Metric": "Low Clutter", "Score": 100 - comparison["ad_a"]["clutter"]},
            {"Ad": "Ad A", "Metric": "Emotion Fit", "Score": comparison["ad_a"]["emotion_score"]},
            {"Ad": "Ad A", "Metric": "Central Saliency", "Score": comparison["ad_a"]["focus_score"]},
            {"Ad": "Ad B", "Metric": "Final Score", "Score": comparison["ad_b"]["final_score"]},
            {"Ad": "Ad B", "Metric": "Low Clutter", "Score": 100 - comparison["ad_b"]["clutter"]},
            {"Ad": "Ad B", "Metric": "Emotion Fit", "Score": comparison["ad_b"]["emotion_score"]},
            {"Ad": "Ad B", "Metric": "Central Saliency", "Score": comparison["ad_b"]["focus_score"]},
        ],
        x="Metric",
        y="Score",
        color="Ad",
        barmode="group",
        range_y=[0, 100],
        title="A/B Creative Performance Model",
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=70, b=30))
    st.plotly_chart(fig, use_container_width=True)


def render_live_metrics(container, metrics: dict) -> None:
    metrics = {
        "face_detected": metrics.get("face_detected", False),
        "surprise": metrics.get("surprise", 0),
        "confusion_anger": metrics.get("confusion_anger", 0),
        "engagement": metrics.get("engagement", 0),
    }
    with container:
        cols = st.columns(4)
        cols[0].metric("Face", "Detected" if metrics["face_detected"] else "Waiting")
        cols[1].metric("Surprise", f"{metrics['surprise']}/100")
        cols[2].metric("Confusion / Anger", f"{metrics['confusion_anger']}/100")
        cols[3].metric("Engagement", f"{metrics['engagement']}/100")


def create_sample_ad(kind: str) -> Image.Image:
    if kind == "cluttered":
        return create_cluttered_sample()
    if kind == "trust":
        return create_trust_sample()
    return create_focused_sample()


def create_focused_sample() -> Image.Image:
    width, height = 1100, 720
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / height
        gradient[y, :, 0] = int(18 + 28 * t)
        gradient[y, :, 1] = int(64 + 55 * (1 - t))
        gradient[y, :, 2] = int(84 + 65 * t)

    image = Image.fromarray(gradient, "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 70, width - 70, height - 70), outline=(255, 255, 255), width=5)
    draw.rounded_rectangle((120, 455, 430, 545), radius=18, fill=(255, 122, 0))
    draw.text((120, 145), "NEUROLENS", fill=(255, 255, 255), font=_font(82))
    draw.text((124, 252), "One message. One action.", fill=(238, 244, 246), font=_font(46))
    draw.text((160, 482), "VIEW OFFER", fill=(18, 30, 46), font=_font(34))
    draw.ellipse((735, 150, 980, 395), fill=(255, 212, 0), outline=(255, 255, 255), width=4)
    draw.arc((760, 190, 950, 360), 200, 340, fill=(18, 30, 46), width=8)
    draw.ellipse((805, 230, 835, 260), fill=(18, 30, 46))
    draw.ellipse((880, 230, 910, 260), fill=(18, 30, 46))
    return image


def create_cluttered_sample() -> Image.Image:
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (720, 1100, 3), dtype=np.uint8)
    image = Image.fromarray(noise, "RGB")
    draw = ImageDraw.Draw(image)
    colors = ["#FF0000", "#FFD400", "#7A35FF", "#00AA44", "#0000FF", "#FF7A00"]
    for i in range(54):
        x = 24 + (i * 131) % 940
        y = 32 + (i * 89) % 610
        fill = colors[i % len(colors)]
        draw.rectangle((x, y, x + 146, y + 58), fill=fill)
        draw.text((x + 8, y + 15), f"DEAL {i + 1}", fill=(255, 255, 255), font=_font(22))
    draw.text((70, 300), "EVERYTHING NOW", fill=(255, 255, 255), font=_font(72))
    return image


def create_trust_sample() -> Image.Image:
    width, height = 1100, 720
    image = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 70, width - 70, height - 70), radius=22, fill="#FFFFFF", outline="#D9E4EA", width=4)
    draw.text((120, 130), "Signal clarity", fill="#10384A", font=_font(72))
    draw.text((124, 220), "See the creative cues your audience feels first.", fill="#315665", font=_font(38))
    draw.rounded_rectangle((120, 485, 425, 565), radius=18, fill="#00AA44")
    draw.text((168, 506), "BOOK DEMO", fill="#FFFFFF", font=_font(32))
    draw.rounded_rectangle((650, 145, 980, 520), radius=20, fill="#E9F4F8", outline="#B7D3DE", width=4)
    draw.rectangle((690, 215, 930, 255), fill="#0000FF")
    draw.rectangle((690, 295, 880, 335), fill="#00AA44")
    draw.rectangle((690, 375, 950, 415), fill="#FFD400")
    draw.ellipse((615, 118, 700, 203), fill="#FF7A00")
    return image


@st.cache_resource(show_spinner=False)
def _font(size: int):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1320px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        .nl-hero {
            border: 1px solid #DCE8EC;
            border-radius: 8px;
            padding: 22px 24px;
            margin-bottom: 18px;
            background: linear-gradient(135deg, #F8FBFC 0%, #EEF8F5 58%, #FFF4E8 100%);
        }
        .nl-hero h1 {
            margin: 4px 0 6px 0;
            color: #102F3B;
            font-size: 2.45rem;
            line-height: 1.1;
            letter-spacing: 0;
        }
        .nl-hero p {
            margin: 0;
            max-width: 850px;
            color: #365966;
            font-size: 1.05rem;
        }
        .nl-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }
        .nl-badges span {
            border: 1px solid #CFE1E6;
            border-radius: 999px;
            background: rgba(255,255,255,.78);
            color: #214654;
            font-size: .82rem;
            font-weight: 700;
            padding: 6px 10px;
        }
        .nl-kicker {
            color: #C75E00;
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .nl-story-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1.4fr;
            gap: 12px;
            margin: 0 0 18px 0;
        }
        .nl-before-after,
        .nl-steps {
            border: 1px solid #DCE8EC;
            border-radius: 8px;
            background: #FFFFFF;
            padding: 14px;
        }
        .nl-after {
            border-color: #AEDCC7;
            background: #F5FBF8;
        }
        .nl-before-after h3 {
            margin: 0 0 7px 0;
            color: #102F3B;
            font-size: 1rem;
            letter-spacing: 0;
        }
        .nl-before-after p {
            margin: 0;
            color: #4D6873;
            font-size: .9rem;
            line-height: 1.35;
        }
        .nl-steps {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .nl-steps div {
            min-width: 0;
        }
        .nl-steps span {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 24px;
            height: 24px;
            border-radius: 999px;
            background: #FF7A00;
            color: #FFFFFF;
            font-size: .8rem;
            font-weight: 800;
        }
        .nl-steps strong {
            display: block;
            margin-top: 7px;
            color: #102F3B;
            font-size: .95rem;
        }
        .nl-steps small {
            display: block;
            color: #58727C;
            line-height: 1.3;
        }
        .nl-insight-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 14px 0 10px 0;
        }
        .nl-insight {
            border: 1px solid #DCE8EC;
            border-radius: 8px;
            padding: 12px;
            background: #FFFFFF;
        }
        .nl-insight span {
            display: block;
            color: #5C7480;
            font-size: .78rem;
        }
        .nl-insight strong {
            display: block;
            color: #102F3B;
            font-size: .96rem;
            margin-top: 3px;
        }
        .nl-swatch {
            height: 44px;
            border-radius: 8px;
            border: 1px solid rgba(80, 92, 104, .35);
            margin-bottom: 8px;
        }
        .nl-contrast-pair {
            display: grid;
            grid-template-columns: 42px minmax(0, 1fr) 42px minmax(0, 1fr);
            gap: 8px;
            align-items: center;
            border: 1px solid #DCE8EC;
            border-radius: 8px;
            padding: 10px;
            margin: 8px 0 12px 0;
            background: #FFFFFF;
        }
        .nl-contrast-pair span {
            display: block;
            width: 42px;
            height: 32px;
            border: 1px solid rgba(80, 92, 104, .35);
            border-radius: 6px;
        }
        .nl-contrast-pair strong {
            min-width: 0;
            color: #102F3B;
            font-size: .9rem;
        }
        @media (max-width: 700px) {
            .nl-insight-row {
                grid-template-columns: 1fr;
            }
            .nl-story-grid {
                grid-template-columns: 1fr;
            }
            .nl-steps {
                grid-template-columns: 1fr;
            }
            .nl-hero h1 {
                font-size: 1.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
