"""
ADX FPM - Passport Photo Processor

A modular, object-oriented passport photo processing system.

Supported formats:
- US Visa (2x2 inch)
- Malaysian Passport (35x50mm)
- UK Passport (35x45mm)
- School ID (40x50mm, blue background)
- JPJ License (25x32mm, 40% face coverage)
- China Visa (33x48mm)
- Japan Visa (35x45mm)
- Umrah / Saudi Visa (4x6cm)

Usage:
    from adxfpm import PhotoProcessor
    processor = PhotoProcessor()
    result = processor.process("input/photo.jpg", "malaysia")
    result.photo.save("output/photo.jpg")
    processor.close()
"""

# Core classes
from .config import PhotoFormat, PhotoFormatRegistry, REGISTRY, DPI, get_format, get_format_list
from .face_detection import FaceDetector, FaceLandmarks, detect_face_landmarks, detect_and_crop_face
from .background import BackgroundRemover, FillStrategy, BackgroundRemovalError, MemoryError, SessionError
from .background import init_rembg_session, remove_background, fill_with_white, fill_with_blue
from .photo_formats import PhotoFormatter
from .sheet import SheetGenerator
from .sheet import (
    create_us_passport_sheet,
    create_my_passport_sheet,
    create_uk_passport_sheet,
    create_school_id_sheet,
    create_jpj_sheet,
)
from .flux_enhance import PhotoEnhancer, FluxEnhanceError, FluxMemoryError, FluxAPIError
from .flux_enhance import enhance_photo, get_pipeline_info
from .utils import GPUInfo, print_gpu_info, print_summary
from .processor import PhotoProcessor, ProcessingResult
