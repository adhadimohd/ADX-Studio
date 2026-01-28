"""
PhotoProcessor — Facade that orchestrates the full passport photo pipeline.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

from PIL import Image

from .config import PhotoFormat, REGISTRY, DPI
from .face_detection import FaceDetector, FaceLandmarks
from .background import BackgroundRemover
from .photo_formats import PhotoFormatter
from .sheet import SheetGenerator
from .flux_enhance import PhotoEnhancer, FluxEnhanceError, FluxMemoryError
from .utils import GPUInfo, SRGB_ICC_BYTES

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of a full photo processing pipeline run."""
    photo: Image.Image
    sheet: Image.Image
    original_photo: Image.Image
    format_info: PhotoFormat
    ai_photo: Optional[Image.Image] = None
    ai_sheet: Optional[Image.Image] = None


class PhotoProcessor:
    """High-level facade for the entire passport photo pipeline.

    Usage:
        processor = PhotoProcessor()
        result = processor.process("input/photo.jpg", "malaysia")
        result.photo.save("output/photo.jpg")
        result.sheet.save("output/sheet.jpg")
        processor.close()
    """

    def __init__(self, force_cpu: bool = False):
        logger.info("Initializing PhotoProcessor...")
        self.face_detector = FaceDetector()
        self.bg_remover = BackgroundRemover(force_cpu=force_cpu)
        self.formatter = PhotoFormatter(self.bg_remover)
        self.sheet_gen = SheetGenerator()
        self.enhancer = PhotoEnhancer()
        logger.info("PhotoProcessor ready")

    def process(
        self,
        input_path: str,
        format_key: str,
        enhance_mode: Optional[str] = None,
        enhance_prompt: Optional[str] = None,
        adjustments: Optional[Dict] = None,
    ) -> ProcessingResult:
        """Run the full pipeline: detect -> crop -> remove bg -> format -> sheet.

        Args:
            input_path: Path to input image.
            format_key: Key from the PhotoFormat registry.
            enhance_mode: Optional AI enhancement mode ('retouch', etc.).
            enhance_prompt: Optional prompt for AI enhancement.

        Returns:
            ProcessingResult with all generated images.

        Raises:
            ValueError: If format_key is not registered.
            Various exceptions from sub-components.
        """
        fmt = REGISTRY.get(format_key)
        if fmt is None:
            raise ValueError(f"Unknown format: {format_key}. Available: {REGISTRY.keys()}")

        logger.info(f"Processing: format={format_key}, input={input_path}")

        # 1. Detect face landmarks
        landmarks = self.face_detector.detect_landmarks(input_path)
        if landmarks is None:
            raise RuntimeError("No face detected in the image")

        # 2. Create formatted photo (includes bg removal internally)
        photo = self.formatter.create_photo(fmt, landmarks, input_path, adjustments=adjustments)

        # 3. Create print sheet
        sheet = self.sheet_gen.create_sheet(photo, fmt)

        # 4. Optional AI enhancement
        ai_photo = None
        ai_sheet = None
        if enhance_mode and enhance_prompt:
            try:
                ai_photo = self.enhancer.enhance(
                    photo, enhance_prompt, mode=enhance_mode,
                    num_inference_steps=4, guidance_scale=3.5,
                )
                ai_photo.info['icc_profile'] = SRGB_ICC_BYTES
                ai_sheet = self.sheet_gen.create_sheet(ai_photo, fmt)
            except (FluxMemoryError, FluxEnhanceError) as e:
                logger.warning(f"AI enhancement skipped: {e}")

        return ProcessingResult(
            photo=photo,
            sheet=sheet,
            original_photo=photo,
            format_info=fmt,
            ai_photo=ai_photo,
            ai_sheet=ai_sheet,
        )

    def reinitialize_bg(self) -> None:
        """Re-create background removal session (recovery after errors)."""
        self.bg_remover.reinitialize()
        self.formatter = PhotoFormatter(self.bg_remover)

    def close(self) -> None:
        """Release all resources."""
        self.bg_remover.close()
        self.enhancer.release()
        logger.info("PhotoProcessor closed")
