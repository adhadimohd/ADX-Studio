"""
Configuration settings for ADX FPM - Passport Photo Processor
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import torch


# =============================================================================
# GENERAL CONFIG
# =============================================================================

INPUT_IMAGE = "input/passport.jpg"
OUTPUT_BASE = "outputs/adxfpm"

os.makedirs(OUTPUT_BASE, exist_ok=True)

DPI = 300
MM_TO_INCH = 1 / 25.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MediaPipe settings
MIN_DETECTION_CONFIDENCE = 0.5
FACE_DETECTOR_MODEL = "models/blaze_face_short_range.tflite"
FACE_PADDING = 0.3


# =============================================================================
# REPLICATE API CONFIG
# =============================================================================

REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN', '')
REPLICATE_MODEL_VERSION = "8e9c42d77b10a2a41af823ac4500f7545be6ebc4e745830fc3f3de10de200542"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

FLUX_MODE = os.environ.get('FLUX_MODE', 'auto')


# =============================================================================
# PHOTO FORMAT DATACLASS & REGISTRY
# =============================================================================

def _calc_print_size(width_mm: float, height_mm: float) -> Tuple[int, int]:
    """Calculate print size in pixels from mm dimensions"""
    return (int(width_mm * MM_TO_INCH * DPI), int(height_mm * MM_TO_INCH * DPI))


def _calc_print_size_inches(width_inch: float, height_inch: float) -> Tuple[int, int]:
    """Calculate print size in pixels from inch dimensions"""
    return (int(width_inch * DPI), int(height_inch * DPI))


@dataclass(frozen=True)
class PhotoFormat:
    """Immutable specification for a passport/ID photo format."""
    key: str
    name: str
    description: str
    size_mm: Optional[Tuple[float, float]]
    size_inches: Tuple[float, float]
    sheet_inches: Tuple[float, float]
    print_size: Tuple[int, int]
    sheet_size: Tuple[int, int]
    bg_color: Tuple[int, int, int]
    file_prefix: str
    face_coverage: float = 0.35
    nose_position: float = 0.50
    mark_length: int = 25
    mark_offset: int = 8

    @property
    def aspect_ratio(self) -> float:
        if self.size_mm:
            return self.size_mm[0] / self.size_mm[1]
        return self.size_inches[0] / self.size_inches[1]


class PhotoFormatRegistry:
    """Registry of all available photo formats."""

    def __init__(self):
        self._formats: Dict[str, PhotoFormat] = {}

    def register(self, fmt: PhotoFormat) -> None:
        self._formats[fmt.key] = fmt

    def get(self, key: str) -> Optional[PhotoFormat]:
        return self._formats.get(key)

    def list_all(self) -> List[PhotoFormat]:
        return list(self._formats.values())

    def keys(self) -> List[str]:
        return list(self._formats.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._formats

    def __iter__(self):
        return iter(self._formats.items())

    def items(self):
        return self._formats.items()


# =============================================================================
# GLOBAL REGISTRY — single source of truth
# =============================================================================

REGISTRY = PhotoFormatRegistry()

REGISTRY.register(PhotoFormat(
    key='us', name='US Visa / Passport', description='2x2 inch',
    size_mm=None, size_inches=(2, 2), sheet_inches=(5, 7),
    print_size=_calc_print_size_inches(2, 2),
    sheet_size=_calc_print_size_inches(5, 7),
    bg_color=(255, 255, 255), file_prefix='us_passport',
    face_coverage=0.45, nose_position=0.50,
    mark_length=30, mark_offset=10,
))

REGISTRY.register(PhotoFormat(
    key='malaysia', name='Malaysian Passport', description='35x50mm',
    size_mm=(35, 50), size_inches=(35 * MM_TO_INCH, 50 * MM_TO_INCH),
    sheet_inches=(4, 6),
    print_size=_calc_print_size(35, 50),
    sheet_size=_calc_print_size_inches(4, 6),
    bg_color=(255, 255, 255), file_prefix='my_passport',
    face_coverage=0.35, nose_position=0.60,
))

REGISTRY.register(PhotoFormat(
    key='uk', name='UK Passport', description='35x45mm',
    size_mm=(35, 45), size_inches=(35 * MM_TO_INCH, 45 * MM_TO_INCH),
    sheet_inches=(4, 6),
    print_size=_calc_print_size(35, 45),
    sheet_size=_calc_print_size_inches(4, 6),
    bg_color=(255, 255, 255), file_prefix='uk_passport',
    face_coverage=0.40, nose_position=0.50,
))

REGISTRY.register(PhotoFormat(
    key='school', name='School ID Photo', description='40x50mm',
    size_mm=(40, 50), size_inches=(40 * MM_TO_INCH, 50 * MM_TO_INCH),
    sheet_inches=(4, 6),
    print_size=_calc_print_size(40, 50),
    sheet_size=_calc_print_size_inches(4, 6),
    bg_color=(0, 127, 255), file_prefix='school_id',
    face_coverage=0.40, nose_position=0.45,
))

REGISTRY.register(PhotoFormat(
    key='jpj', name='JPJ License Photo', description='25x32mm',
    size_mm=(25, 32), size_inches=(25 * MM_TO_INCH, 32 * MM_TO_INCH),
    sheet_inches=(4, 6),
    print_size=_calc_print_size(25, 32),
    sheet_size=_calc_print_size_inches(4, 6),
    bg_color=(255, 255, 255), file_prefix='jpj_license',
    face_coverage=0.40, nose_position=0.40,
    mark_length=15, mark_offset=5,
))

REGISTRY.register(PhotoFormat(
    key='china', name='China Visa', description='33x48mm',
    size_mm=(33, 48), size_inches=(33 * MM_TO_INCH, 48 * MM_TO_INCH),
    sheet_inches=(4, 6),
    print_size=_calc_print_size(33, 48),
    sheet_size=_calc_print_size_inches(4, 6),
    bg_color=(255, 255, 255), file_prefix='china_visa',
    face_coverage=0.45, nose_position=0.45,
))

REGISTRY.register(PhotoFormat(
    key='japan', name='Japan Visa', description='35x45mm',
    size_mm=(35, 45), size_inches=(35 * MM_TO_INCH, 45 * MM_TO_INCH),
    sheet_inches=(4, 6),
    print_size=_calc_print_size(35, 45),
    sheet_size=_calc_print_size_inches(4, 6),
    bg_color=(255, 255, 255), file_prefix='japan_visa',
    face_coverage=0.48, nose_position=0.57,
))

REGISTRY.register(PhotoFormat(
    key='umrah', name='Umrah / Saudi Visa', description='4x6cm',
    size_mm=(40, 60), size_inches=(40 * MM_TO_INCH, 60 * MM_TO_INCH),
    sheet_inches=(5, 7),
    print_size=_calc_print_size(40, 60),
    sheet_size=_calc_print_size_inches(5, 7),
    bg_color=(255, 255, 255), file_prefix='umrah_visa',
    face_coverage=0.40, nose_position=0.50,
))


# =============================================================================
# HELPER — for web.py format list API
# =============================================================================

def get_format(format_key: str) -> Optional[PhotoFormat]:
    return REGISTRY.get(format_key)


def get_format_list() -> List[dict]:
    return [
        {'key': fmt.key, 'name': fmt.name, 'description': fmt.description,
         'sheet_inches': list(fmt.sheet_inches)}
        for fmt in REGISTRY.list_all()
    ]


# =============================================================================
# LEGACY EXPORTS (kept for any external code that imports these directly)
# =============================================================================

CUTOUT_OUTPUT = f"{OUTPUT_BASE}/cutout.png"
FACE_ONLY_OUTPUT = f"{OUTPUT_BASE}/face_only.jpg"
SCHOOL_ID_BG_COLOR = REGISTRY.get('school').bg_color
