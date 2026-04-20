"""Live webcam frame callbacks for the neuromarketing lab.

The processor is deliberately stateless from a product perspective: it keeps
only the latest frame's telemetry in memory so Streamlit can display live
metrics without a database or durable session history.
"""

from __future__ import annotations

import threading

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase


class EmotionVideoProcessor(VideoProcessorBase):
    """OpenCV frame processor that estimates simple emotional telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics = {
            "face_detected": False,
            "surprise": 0,
            "confusion_anger": 0,
            "engagement": 0,
        }
        self._face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

        metrics = {
            "face_detected": False,
            "surprise": 0,
            "confusion_anger": 0,
            "engagement": 0,
        }

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_gray = gray[y : y + h, x : x + w]
            metrics = self._estimate_emotions(face_gray)
            metrics["face_detected"] = True

            color = (0, 220, 140)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"S:{metrics['surprise']} C:{metrics['confusion_anger']} E:{metrics['engagement']}"
            cv2.putText(image, label, (x, max(24, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        with self._lock:
            self._metrics = metrics

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def get_metrics(self) -> dict:
        """Return the latest frame metrics for Streamlit metric cards."""
        with self._lock:
            return dict(self._metrics)

    def _estimate_emotions(self, face_gray: np.ndarray) -> dict:
        """Convert simple facial geometry proxies into emotion metrics.

        This is intentionally a lightweight heuristic, not clinical affect
        recognition. The face is split into upper, eye, brow, and mouth bands.
        Wide/detected eyes plus an open dark mouth region are treated as a
        surprise proxy. Dense vertical edges and dark compression in the brow
        band are treated as a confusion/anger proxy. Mouth-region edge activity
        and openness are treated as engagement because smiles, speech, and open
        mouths all increase contrast in that part of the face.
        """
        resized = cv2.resize(face_gray, (160, 160), interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)

        eye_band = equalized[38:82, 18:142]
        brow_band = equalized[24:58, 22:138]
        mouth_band = equalized[98:145, 32:128]

        eyes = self._eye_detector.detectMultiScale(eye_band, scaleFactor=1.1, minNeighbors=4, minSize=(18, 12))
        eye_presence = min(len(eyes), 2) / 2

        mouth_dark_ratio = float(np.mean(mouth_band < np.percentile(mouth_band, 38)))
        mouth_edges = cv2.Canny(mouth_band, 50, 140)
        mouth_edge_ratio = float(np.count_nonzero(mouth_edges) / mouth_edges.size)

        brow_edges = cv2.Sobel(brow_band, cv2.CV_64F, 1, 0, ksize=3)
        brow_tension = float(np.mean(np.abs(brow_edges)) / 255)
        brow_darkness = float(1 - (np.mean(brow_band) / 255))

        surprise = np.clip((eye_presence * 52) + (mouth_dark_ratio * 48), 0, 100)
        confusion_anger = np.clip((brow_tension * 170) + (brow_darkness * 38), 0, 100)
        engagement = np.clip((mouth_edge_ratio * 320) + (mouth_dark_ratio * 42), 0, 100)

        return {
            "surprise": int(round(surprise)),
            "confusion_anger": int(round(confusion_anger)),
            "engagement": int(round(engagement)),
        }
