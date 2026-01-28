"""
Print sheet creation module
"""

import logging
from PIL import Image, ImageDraw

from .config import DPI, PhotoFormat, REGISTRY
from .utils import SRGB_ICC_BYTES

logger = logging.getLogger(__name__)


class SheetGenerator:
    """Generates printable photo sheets for any PhotoFormat."""

    def create_sheet(self, photo: Image.Image, fmt: PhotoFormat) -> Image.Image:
        """Create a 2x2 grid print sheet.

        Args:
            photo: The passport photo to tile.
            fmt: PhotoFormat with print_size, sheet_size, etc.

        Returns:
            PIL Image of the print sheet.
        """
        print_size = fmt.print_size
        sheet_size = fmt.sheet_size
        sheet_inches = fmt.sheet_inches
        photo_mm = fmt.size_mm

        print(f"\nCreating {fmt.name} photo sheet ({sheet_inches[0]}x{sheet_inches[1]} inch, {DPI} DPI)...")

        photo_resized = photo.resize(print_size, Image.LANCZOS)
        if photo_mm:
            print(f"  Photo resized to: {print_size[0]}x{print_size[1]}px ({photo_mm[0]}x{photo_mm[1]}mm @ {DPI} DPI)")
        else:
            print(f"  Photo resized to: {print_size[0]}x{print_size[1]}px @ {DPI} DPI")

        sheet = Image.new('RGB', sheet_size, 'white')
        print(f"  Sheet size: {sheet_size[0]}x{sheet_size[1]}px ({sheet_inches[0]}x{sheet_inches[1]} inch @ {DPI} DPI)")

        total_photos_width = print_size[0] * 2
        total_photos_height = print_size[1] * 2

        start_x = (sheet_size[0] - total_photos_width) // 2
        start_y = (sheet_size[1] - total_photos_height) // 2

        positions = [
            (start_x, start_y),
            (start_x + print_size[0], start_y),
            (start_x, start_y + print_size[1]),
            (start_x + print_size[0], start_y + print_size[1]),
        ]

        for i, (x, y) in enumerate(positions):
            sheet.paste(photo_resized, (x, y))
            print(f"  Photo {i+1} placed at: ({x}, {y})")

        draw = ImageDraw.Draw(sheet)
        print("  Adding cutting marks...")

        for (x, y) in positions:
            self._draw_cutting_marks(
                draw, x, y, print_size[0], print_size[1],
                mark_length=fmt.mark_length, mark_offset=fmt.mark_offset,
            )

        print(f"{fmt.name} sheet created with 4 photos (2x2 grid) and cutting marks")
        sheet.info['icc_profile'] = SRGB_ICC_BYTES
        return sheet

    @staticmethod
    def _draw_cutting_marks(
        draw: ImageDraw.Draw,
        x: int, y: int, width: int, height: int,
        mark_length: int = 25, mark_offset: int = 8,
        color: str = 'black', line_width: int = 2,
    ) -> None:
        """Draw L-shaped cutting marks at the corners of a rectangle."""
        # Top-left
        draw.line([(x - mark_offset - mark_length, y), (x - mark_offset, y)], fill=color, width=line_width)
        draw.line([(x, y - mark_offset - mark_length), (x, y - mark_offset)], fill=color, width=line_width)
        # Top-right
        draw.line([(x + width + mark_offset, y), (x + width + mark_offset + mark_length, y)], fill=color, width=line_width)
        draw.line([(x + width, y - mark_offset - mark_length), (x + width, y - mark_offset)], fill=color, width=line_width)
        # Bottom-left
        draw.line([(x - mark_offset - mark_length, y + height), (x - mark_offset, y + height)], fill=color, width=line_width)
        draw.line([(x, y + height + mark_offset), (x, y + height + mark_offset + mark_length)], fill=color, width=line_width)
        # Bottom-right
        draw.line([(x + width + mark_offset, y + height), (x + width + mark_offset + mark_length, y + height)], fill=color, width=line_width)
        draw.line([(x + width, y + height + mark_offset), (x + width, y + height + mark_offset + mark_length)], fill=color, width=line_width)


# =============================================================================
# Module-level convenience functions (backward compatibility)
# =============================================================================

_default_generator = SheetGenerator()


def _sheet_for(key: str):
    """Create a backward-compatible sheet function for a format key."""
    def _fn(photo_img):
        fmt = REGISTRY.get(key)
        return _default_generator.create_sheet(photo_img, fmt)
    return _fn


# Legacy named functions — all delegate to SheetGenerator
create_us_passport_sheet = _sheet_for('us')
create_my_passport_sheet = _sheet_for('malaysia')
create_uk_passport_sheet = _sheet_for('uk')
create_school_id_sheet = _sheet_for('school')
create_jpj_sheet = _sheet_for('jpj')
create_china_visa_sheet = _sheet_for('china')
create_japan_visa_sheet = _sheet_for('japan')
create_umrah_visa_sheet = _sheet_for('umrah')

# Also keep the standalone helper
draw_cutting_marks = SheetGenerator._draw_cutting_marks
