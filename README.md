# NeuroLens AI: Creative Response Lab

NeuroLens AI is a stateless Streamlit application for creative audits, mock A/B prediction, and live webcam-based biometric telemetry. It uses computer vision and lightweight analytics to help evaluate digital advertising assets without a database or persistent storage layer.

## Architecture

- `app.py`: Streamlit UI, tabs, sample creatives, upload handling, charts, metrics, and webcam UI.
- `analytics.py`: Shannon entropy, clutter scoring, dominant color extraction, color psychology mapping, saliency scoring, and heatmap generation.
- `webrtc_callbacks.py`: OpenCV frame-by-frame webcam processing for basic facial telemetry.

All analysis is computed in memory. Streamlit cache usage is ephemeral and only avoids repeating deterministic image calculations during the running process.

## Features

- **Visual Asset Audit**: Upload or select a sample ad, calculate visual clutter from Shannon entropy, extract top colors with K-Means, map colors to psychology labels, tune the heatmap overlay, and download JSON/PNG outputs.
- **Accessibility & Platform Risk Scanner**: Estimate WCAG contrast risk, flag low-contrast visual regions, and provide non-destructive remediation guidance for ad creative.
- **Creative Doctor**: Generate a cleaner recommended mock layout, compare before/after score deltas, and download the recommended PNG plus JSON report.
- **A/B Predictor**: Compare two ads with deterministic mock scoring based on clutter, target emotion fit, and central saliency, then download a JSON prediction report with confidence labeling.
- **Campaign Simulator**: Adjust clutter, saliency, and dominant color psychology assumptions to forecast mock CPC/CVR outcomes and persona fit in real time.
- **Neuromarketing Lab**: Show an ad stimulus beside a `streamlit-webrtc` webcam feed, display predicted response metrics, and compare them with latest-frame Surprise, Confusion/Anger, Engagement, and face-detection telemetry.

## Demo Flow

1. Open the app and leave **Sample** selected.
2. Review the default **Focused CTA** audit and download the audit report.
3. Open **Creative Doctor** to generate a simplified recommended version of the cluttered sample.
4. Open **A/B Predictor** to compare the clean sample against the cluttered retail sample.
5. Open **Neuromarketing Lab**, start the camera if available, and observe the live telemetry placeholders or camera-denial fallback.

The webcam emotion metrics are heuristic demo signals. They are not clinical affect recognition and should not be used for sensitive decisions.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

You can also use:

```bash
make install
make run
```

## Test

```bash
.venv/bin/python -m unittest discover -s tests
```

Or run all local checks:

```bash
make check
```

## Docker

```bash
docker build -t neurolens-ai .
docker run --rm -p 8501:8501 neurolens-ai
```

## Notes

- Webcam access works best on `localhost` or HTTPS.
- If camera permission is denied, the lab keeps the stimulus and neutral metrics visible.
- Supported upload formats are JPG, JPEG, and PNG.
- `.streamlit/config.toml` defines the app theme and upload limit for deployment-friendly defaults.
