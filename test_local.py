#!/usr/bin/env python3
"""
Local test script for Malaysia Passport Photo Processing
"""

import warnings
import os

os.environ['ORT_LOGGING_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from adxfpm import PhotoProcessor, GPUInfo, DPI


def test_malaysia_passport():
    """Test Malaysia passport photo processing using PhotoProcessor."""

    print("=" * 70)
    print("MALAYSIA PASSPORT PHOTO PROCESSOR - LOCAL TEST")
    print("=" * 70)

    input_image = "input/passport.jpg"
    passport_output = "outputs/test_my_passport.jpg"
    sheet_output = "outputs/test_my_passport_sheet.jpg"

    os.makedirs("outputs", exist_ok=True)

    GPUInfo.print_info()

    if not os.path.exists(input_image):
        print(f"\n[ERROR] Could not find {input_image}")
        print("   Please place passport.jpg in the input folder")
        return False

    print(f"\n[OK] Input image found: {input_image}")

    try:
        processor = PhotoProcessor()

        print("\n" + "=" * 70)
        print("PROCESSING: MALAYSIAN PASSPORT")
        print("=" * 70)

        result = processor.process(input_image, "malaysia")

        # Save outputs
        result.photo.save(passport_output, 'JPEG', quality=95, dpi=(DPI, DPI))
        photo_size = os.path.getsize(passport_output) / 1024**2
        print(f"[OK] Malaysia passport photo saved: {passport_output}")
        print(f"  Size: {result.photo.size}, File: {photo_size:.2f} MB")

        result.sheet.save(sheet_output, 'JPEG', quality=95, dpi=(DPI, DPI))
        sheet_size = os.path.getsize(sheet_output) / 1024**2
        print(f"[OK] Malaysia passport sheet saved: {sheet_output}")
        print(f"  Size: {result.sheet.size}, File: {sheet_size:.2f} MB")
        print(f"  DPI: {DPI}")

        processor.close()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  1. Malaysia Passport: {passport_output}")
    print(f"  2. Malaysia Passport Sheet: {sheet_output}")
    print("\n[OK] All tests passed!")

    return True


if __name__ == "__main__":
    success = test_malaysia_passport()
    exit(0 if success else 1)
