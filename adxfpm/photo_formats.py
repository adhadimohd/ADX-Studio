"""
Photo format processor — generic, data-driven by PhotoFormat.
"""

import logging
from typing import Optional

from PIL import Image, ImageEnhance
from PIL.ImageCms import profileToProfile, createProfile, ImageCmsProfile

from .config import PhotoFormat, INPUT_IMAGE, DPI
from .face_detection import FaceLandmarks
from .background import BackgroundRemover, FillStrategy
from .utils import SRGB_ICC_BYTES

_srgb_profile = ImageCmsProfile(createProfile('sRGB'))

logger = logging.getLogger(__name__)


class PhotoFormatter:
    """Creates passport/ID photos for any format using landmark-based cropping."""

    def __init__(self, bg_remover: BackgroundRemover):
        self._bg_remover = bg_remover

    def create_photo(
        self,
        fmt: PhotoFormat,
        landmarks: Optional[FaceLandmarks],
        input_path: str = INPUT_IMAGE,
        fallback_img: Optional[Image.Image] = None,
        adjustments: Optional[dict] = None,
    ) -> Image.Image:
        """Create a photo for the given format.

        Args:
            fmt: PhotoFormat specification.
            landmarks: FaceLandmarks (or None for fallback crop).
            input_path: Path to the original input image.
            fallback_img: Pre-cropped image to use if no landmarks.
            adjustments: Optional dict with brightness, rotation, face_scale, nose_offset.

        Returns:
            Final PIL Image at the format's print_size.
        """
        adj = adjustments or {}
        print(f"\n{'='*70}")
        print(f"CREATE {fmt.name.upper()} ({fmt.description})")
        print(f"{'='*70}")

        if landmarks:
            photo = self._crop_by_landmarks(fmt, landmarks, input_path, adj)
        elif fallback_img is not None:
            print("  No face landmarks, using center crop fallback")
            photo = self._fallback_crop(fallback_img, fmt.aspect_ratio)
        else:
            raise ValueError("Either landmarks or fallback_img must be provided")

        print(f"  Cropped to {fmt.description} ratio: {photo.width}x{photo.height}px")

        final = photo.resize(fmt.print_size, Image.LANCZOS)

        # Apply rotation
        rotation = adj.get('rotation', 0)
        if rotation:
            final = final.rotate(rotation, resample=Image.BICUBIC, expand=False, fillcolor=fmt.bg_color)

        # Apply brightness
        brightness = adj.get('brightness', 0)
        if brightness:
            final = ImageEnhance.Brightness(final).enhance(1 + brightness / 100)

        # Embed sRGB ICC profile
        final.info['icc_profile'] = SRGB_ICC_BYTES

        return final

    def _crop_by_landmarks(
        self,
        fmt: PhotoFormat,
        landmarks: FaceLandmarks,
        input_path: str,
        adj: Optional[dict] = None,
    ) -> Image.Image:
        """Crop image based on face landmarks and format spec."""
        adj = adj or {}
        face_height = landmarks.face_height
        nose_x, nose_y = landmarks.nose_tip
        forehead_y = landmarks.forehead[1]
        chin_y = landmarks.chin[1]
        orig_img_width, orig_img_height = landmarks.image_size

        # Apply face_scale adjustment: positive = bigger face = smaller coverage
        face_scale = adj.get('face_scale', 0)
        effective_coverage = fmt.face_coverage * (1 - face_scale / 100)
        effective_coverage = max(0.15, min(0.90, effective_coverage))

        crop_height = int(face_height / effective_coverage)
        crop_width = int(crop_height * fmt.aspect_ratio)

        # Apply nose_offset adjustment: shift crop vertically
        nose_offset_px = int(face_height * adj.get('nose_offset', 0) / 100)

        # JPJ uses face center, others use nose position
        if 'jpj' in fmt.key:
            face_center_y = (forehead_y + chin_y) // 2
            crop_y1 = face_center_y - int(crop_height * fmt.nose_position) + nose_offset_px
        else:
            crop_y1 = nose_y - int(crop_height * fmt.nose_position) + nose_offset_px

        crop_x1 = nose_x - crop_width // 2
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        # Adjust bounds
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_x2 > orig_img_width:
            crop_x1 -= (crop_x2 - orig_img_width)
            crop_x2 = orig_img_width
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_y2 > orig_img_height:
            crop_y1 -= (crop_y2 - orig_img_height)
            crop_y2 = orig_img_height

        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(orig_img_width, crop_x2)
        crop_y2 = min(orig_img_height, crop_y2)

        orig_img = Image.open(input_path)
        # Convert to sRGB if the image has a different ICC profile
        src_profile = orig_img.info.get('icc_profile')
        if src_profile:
            try:
                orig_img = profileToProfile(orig_img, src_profile, _srgb_profile, outputMode='RGB')
            except Exception:
                orig_img = orig_img.convert('RGB')
        else:
            orig_img = orig_img.convert('RGB')
        crop = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        print(f"  Cropping: ({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})")
        print(f"  Crop size: {crop_x2-crop_x1}x{crop_y2-crop_y1}px")
        print(f"  Face height: {face_height}px, Target coverage: {fmt.face_coverage*100:.0f}%")

        cutout = self._bg_remover.remove(crop)
        photo = self._bg_remover.fill(cutout, fmt.bg_color)

        actual_coverage = face_height / (crop_y2 - crop_y1) * 100
        print(f"  Actual face coverage: {actual_coverage:.1f}%")

        return photo

    @staticmethod
    def _fallback_crop(img: Image.Image, target_ratio: float) -> Image.Image:
        """Fallback center crop when no landmarks available."""
        src_width, src_height = img.size
        src_ratio = src_width / src_height

        if src_ratio > target_ratio:
            new_width = int(src_height * target_ratio)
            crop_x = (src_width - new_width) // 2
            crop_box = (crop_x, 0, crop_x + new_width, src_height)
        else:
            new_height = int(src_width / target_ratio)
            crop_y = (src_height - new_height) // 2
            crop_box = (0, crop_y, src_width, crop_y + new_height)

        return img.crop(crop_box)
