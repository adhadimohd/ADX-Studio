#!/usr/bin/env python3
"""
ADX FPM - Passport Photo Processor
Main entry point
"""

import warnings
import os

# Suppress warnings
os.environ['ORT_LOGGING_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from .config import (
    INPUT_IMAGE, DPI, CUTOUT_OUTPUT,
    US_PASSPORT_OUTPUT, US_PASSPORT_SHEET_OUTPUT, US_SHEET_INCHES, US_SHEET_SIZE,
    MY_PASSPORT_OUTPUT, MY_PASSPORT_SHEET_OUTPUT, MY_SHEET_INCHES, MY_SHEET_SIZE,
    UK_PASSPORT_OUTPUT, UK_PASSPORT_SHEET_OUTPUT, UK_SHEET_INCHES, UK_SHEET_SIZE,
    SCHOOL_ID_OUTPUT, SCHOOL_ID_SHEET_OUTPUT, SCHOOL_ID_SHEET_INCHES, SCHOOL_ID_SHEET_SIZE,
    JPJ_OUTPUT, JPJ_SHEET_OUTPUT, JPJ_SHEET_INCHES, JPJ_SHEET_SIZE,
    FACE_ONLY_OUTPUT
)
from .face_detection import detect_face_landmarks, detect_and_crop_face
from .background import init_rembg_session, remove_background, fill_with_white
from .sheet import (
    create_us_passport_sheet,
    create_my_passport_sheet,
    create_uk_passport_sheet,
    create_school_id_sheet,
    create_jpj_sheet
)
from .photo_formats import (
    create_us_passport,
    create_my_passport,
    create_uk_passport,
    create_school_id,
    create_jpj_license
)
from .utils import print_gpu_info, print_summary


def main():
    print("="*70)
    print("PASSPORT PHOTO PROCESSOR")
    print("Face detection -> Crop -> Background removal -> Multi-format output")
    print("="*70)

    print_gpu_info()

    # Check input
    if not os.path.exists(INPUT_IMAGE):
        print(f"\nError: Could not find {INPUT_IMAGE}")
        print(f"   Please place your image in the input folder")
        return

    print(f"\nInput: {INPUT_IMAGE}")

    # Initialize models
    print("\nInitializing models...")
    rembg_session = init_rembg_session()

    # Step 1: Detect face landmarks
    print("\n" + "="*70)
    print("STEP 1: DETECT FACE AND LANDMARKS")
    print("="*70)

    face_landmarks = detect_face_landmarks(INPUT_IMAGE)
    face_crop = detect_and_crop_face(INPUT_IMAGE)

    if face_crop is None:
        print("\nCannot proceed without face detection")
        return

    # Step 2: Remove background
    print("\n" + "="*70)
    print("STEP 2: REMOVE BACKGROUND")
    print("="*70)

    cutout = remove_background(face_crop, rembg_session)
    cutout.save(CUTOUT_OUTPUT, format='PNG', optimize=True)
    cutout_size = os.path.getsize(CUTOUT_OUTPUT) / 1024**2
    print(f"Saved cutout: {CUTOUT_OUTPUT} ({cutout_size:.1f}MB)")

    # Step 3: Create base passport image
    print("\n" + "="*70)
    print("STEP 3: CREATE BASE PASSPORT IMAGE")
    print("="*70)

    passport_img = fill_with_white(cutout)

    # Create all photo formats
    outputs = {
        'face_only': FACE_ONLY_OUTPUT,
        'cutout': CUTOUT_OUTPUT,
    }

    # US Visa
    final_us = create_us_passport(passport_img)
    us_sheet = create_us_passport_sheet(final_us)
    us_sheet.save(US_PASSPORT_SHEET_OUTPUT, 'JPEG', quality=95, dpi=(DPI, DPI))
    print(f"Saved US visa sheet: {US_PASSPORT_SHEET_OUTPUT}")
    print(f"  Size: {US_SHEET_SIZE[0]}x{US_SHEET_SIZE[1]}px ({US_SHEET_INCHES[0]}x{US_SHEET_INCHES[1]} inch @ {DPI} DPI)")
    outputs['us_passport'] = US_PASSPORT_OUTPUT
    outputs['us_passport_sheet'] = US_PASSPORT_SHEET_OUTPUT

    # Malaysian Passport
    final_my = create_my_passport(passport_img, face_landmarks, rembg_session)
    my_sheet = create_my_passport_sheet(final_my)
    my_sheet.save(MY_PASSPORT_SHEET_OUTPUT, 'JPEG', quality=95, dpi=(DPI, DPI))
    print(f"Saved Malaysian passport sheet: {MY_PASSPORT_SHEET_OUTPUT}")
    print(f"  Size: {MY_SHEET_SIZE[0]}x{MY_SHEET_SIZE[1]}px ({MY_SHEET_INCHES[0]}x{MY_SHEET_INCHES[1]} inch @ {DPI} DPI)")
    outputs['my_passport'] = MY_PASSPORT_OUTPUT
    outputs['my_passport_sheet'] = MY_PASSPORT_SHEET_OUTPUT

    # UK Passport
    final_uk = create_uk_passport(passport_img, face_landmarks, rembg_session)
    uk_sheet = create_uk_passport_sheet(final_uk)
    uk_sheet.save(UK_PASSPORT_SHEET_OUTPUT, 'JPEG', quality=95, dpi=(DPI, DPI))
    print(f"Saved UK passport sheet: {UK_PASSPORT_SHEET_OUTPUT}")
    print(f"  Size: {UK_SHEET_SIZE[0]}x{UK_SHEET_SIZE[1]}px ({UK_SHEET_INCHES[0]}x{UK_SHEET_INCHES[1]} inch @ {DPI} DPI)")
    outputs['uk_passport'] = UK_PASSPORT_OUTPUT
    outputs['uk_passport_sheet'] = UK_PASSPORT_SHEET_OUTPUT

    # School ID
    final_school = create_school_id(passport_img, face_landmarks, rembg_session)
    school_sheet = create_school_id_sheet(final_school)
    school_sheet.save(SCHOOL_ID_SHEET_OUTPUT, 'JPEG', quality=95, dpi=(DPI, DPI))
    print(f"Saved School ID sheet: {SCHOOL_ID_SHEET_OUTPUT}")
    print(f"  Size: {SCHOOL_ID_SHEET_SIZE[0]}x{SCHOOL_ID_SHEET_SIZE[1]}px ({SCHOOL_ID_SHEET_INCHES[0]}x{SCHOOL_ID_SHEET_INCHES[1]} inch @ {DPI} DPI)")
    outputs['school_id'] = SCHOOL_ID_OUTPUT
    outputs['school_id_sheet'] = SCHOOL_ID_SHEET_OUTPUT

    # JPJ License
    final_jpj = create_jpj_license(passport_img, face_landmarks, rembg_session)
    jpj_sheet = create_jpj_sheet(final_jpj)
    jpj_sheet.save(JPJ_SHEET_OUTPUT, 'JPEG', quality=95, dpi=(DPI, DPI))
    print(f"Saved JPJ license sheet: {JPJ_SHEET_OUTPUT}")
    print(f"  Size: {JPJ_SHEET_SIZE[0]}x{JPJ_SHEET_SIZE[1]}px ({JPJ_SHEET_INCHES[0]}x{JPJ_SHEET_INCHES[1]} inch @ {DPI} DPI)")
    outputs['jpj_license'] = JPJ_OUTPUT
    outputs['jpj_license_sheet'] = JPJ_SHEET_OUTPUT

    # Print summary
    print_summary(outputs)
    print_gpu_info()


if __name__ == "__main__":
    main()
