"""
Face detection module using MediaPipe
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from PIL import Image

from .config import MIN_DETECTION_CONFIDENCE, FACE_DETECTOR_MODEL, FACE_ONLY_OUTPUT

logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """Detected face landmark positions and measurements."""
    nose_tip: Tuple[int, int]
    forehead: Tuple[int, int]
    chin: Tuple[int, int]
    right_ear: Tuple[int, int]
    left_ear: Tuple[int, int]
    image_size: Tuple[int, int]
    face_height: int
    face_width: int


class FaceDetector:
    """Detects faces and landmarks using MediaPipe."""

    def __init__(
        self,
        min_confidence: float = MIN_DETECTION_CONFIDENCE,
        detector_model: str = FACE_DETECTOR_MODEL,
        landmarker_model: str = "models/face_landmarker.task",
    ):
        self._min_confidence = min_confidence
        self._detector_model = detector_model
        self._landmarker_model = landmarker_model

    def _ensure_landmarker_model(self) -> None:
        if not os.path.exists(self._landmarker_model):
            print(f"Face landmarker model not found at {self._landmarker_model}")
            print("  Downloading model...")
            import urllib.request
            os.makedirs(os.path.dirname(self._landmarker_model) or "models", exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self._landmarker_model)
            print(f"  Model downloaded to {self._landmarker_model}")

    def detect_landmarks(self, image_path: str) -> Optional[FaceLandmarks]:
        """Detect face landmarks using MediaPipe Face Landmarker.

        Returns:
            FaceLandmarks dataclass or None if no face found.
        """
        print("Detecting face landmarks with MediaPipe Face Landmarker...")

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print("Failed to load image")
            return None

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        self._ensure_landmarker_model()

        base_options = mp.tasks.BaseOptions(model_asset_path=self._landmarker_model)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=self._min_confidence,
            min_face_presence_confidence=self._min_confidence,
            min_tracking_confidence=self._min_confidence,
        )

        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        detection_result = landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            print("No face landmarks detected!")
            landmarker.close()
            return None

        face_lms = detection_result.face_landmarks[0]

        nose_tip = face_lms[4]
        forehead = face_lms[10]
        chin = face_lms[152]
        right_ear = face_lms[234]
        left_ear = face_lms[454]

        landmarks = FaceLandmarks(
            nose_tip=(int(nose_tip.x * img_width), int(nose_tip.y * img_height)),
            forehead=(int(forehead.x * img_width), int(forehead.y * img_height)),
            chin=(int(chin.x * img_width), int(chin.y * img_height)),
            right_ear=(int(right_ear.x * img_width), int(right_ear.y * img_height)),
            left_ear=(int(left_ear.x * img_width), int(left_ear.y * img_height)),
            image_size=(img_width, img_height),
            face_height=int(chin.y * img_height) - int(forehead.y * img_height),
            face_width=int(left_ear.x * img_width) - int(right_ear.x * img_width),
        )

        print(f"Face landmarks detected:")
        print(f"  Nose tip: {landmarks.nose_tip}")
        print(f"  Forehead: {landmarks.forehead}")
        print(f"  Chin: {landmarks.chin}")
        print(f"  Face height: {landmarks.face_height}px")
        print(f"  Face width: {landmarks.face_width}px")

        landmarker.close()
        return landmarks

    def detect_and_crop(self, image_path: str) -> Optional[Image.Image]:
        """Detect face and return cropped face area with padding."""
        print("Detecting face with MediaPipe...")

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print("Failed to load image")
            return None

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        base_options = mp.tasks.BaseOptions(model_asset_path=self._detector_model)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self._min_confidence,
        )

        detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        detection_result = detector.detect(mp_image)

        if not detection_result.detections:
            print("No face detected!")
            return None

        detection = detection_result.detections[0]
        bbox = detection.bounding_box

        x1 = bbox.origin_x
        y1 = bbox.origin_y
        width = bbox.width
        height = bbox.height
        x2 = x1 + width
        y2 = y1 + height

        padding_x = int(width * 0.80)
        padding_y = int(height * 0.80)

        x1_pad = max(0, x1 - padding_x)
        y1_pad = max(0, y1 - padding_y)
        x2_pad = min(img_width, x2 + padding_x)
        y2_pad = min(img_height, y2 + padding_y)

        # Save face only (no padding)
        face_only_cv = img_rgb[y1:y2, x1:x2]
        face_only = Image.fromarray(face_only_cv)
        face_only.save(FACE_ONLY_OUTPUT, 'JPEG', quality=95)
        print(f"Saved face only (no padding): {FACE_ONLY_OUTPUT}")
        print(f"  Face size: {face_only.width}x{face_only.height}px")

        # Crop face with padding
        face_crop_cv = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
        face_crop = Image.fromarray(face_crop_cv)

        confidence = detection.categories[0].score
        print(f"Face detected (confidence: {confidence:.2f})")
        print(f"  Location: ({x1_pad}, {y1_pad}, {x2_pad}, {y2_pad})")
        print(f"  Original size: {img_width}x{img_height}")
        print(f"  Cropped size (with padding): {face_crop.width}x{face_crop.height}")

        detector.close()
        return face_crop


# =============================================================================
# Module-level convenience functions (backward compatibility)
# =============================================================================

_default_detector = None


def _get_default_detector() -> FaceDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector()
    return _default_detector


def detect_face_landmarks(image_path: str) -> Optional[FaceLandmarks]:
    return _get_default_detector().detect_landmarks(image_path)


def detect_and_crop_face(image_path: str) -> Optional[Image.Image]:
    return _get_default_detector().detect_and_crop(image_path)
