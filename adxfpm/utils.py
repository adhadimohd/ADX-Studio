"""
Utility functions
"""

from typing import Dict, Optional
import torch
from PIL.ImageCms import createProfile, ImageCmsProfile

# Build the sRGB ICC profile once (bytes), for embedding in saved images
_srgb_profile = ImageCmsProfile(createProfile('sRGB'))
SRGB_ICC_BYTES = _srgb_profile.tobytes()


class GPUInfo:
    """GPU information and diagnostics."""

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_info() -> Dict:
        """Return GPU info as a dict."""
        if not torch.cuda.is_available():
            return {'available': False, 'device': 'CPU'}

        props = torch.cuda.get_device_properties(0)
        info = {
            'available': True,
            'device': 'CUDA',
            'name': torch.cuda.get_device_name(0),
            'vram_total_gb': round(props.total_memory / 1024**3, 1),
            'vram_allocated_gb': round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            'vram_cached_gb': round(torch.cuda.memory_reserved(0) / 1024**3, 2),
        }
        try:
            free_mem = torch.cuda.mem_get_info()[0]
            info['vram_free_gb'] = round(free_mem / 1024**3, 1)
        except Exception:
            pass
        return info

    @staticmethod
    def print_info() -> None:
        """Print GPU information to stdout."""
        info = GPUInfo.get_info()
        if info['available']:
            print(f"GPU: {info['name']}")
            print(f"VRAM Total: {info['vram_total_gb']} GB")
            print(f"VRAM Allocated: {info['vram_allocated_gb']} GB")
            print(f"VRAM Cached: {info['vram_cached_gb']} GB")
        else:
            print("No CUDA GPU detected")


# Backward compatibility
def print_gpu_info():
    GPUInfo.print_info()


def print_summary(outputs):
    """Print final summary of all output files"""
    print("\n" + "="*70)
    print("SUCCESS! ALL PASSPORT PHOTOS CREATED")
    print("="*70)
    print(f"\nOutput files:")

    categories = [
        ("Base outputs", [
            ("face_only", "Detected face only (no padding)"),
            ("cutout", "Transparent PNG with background removed"),
        ]),
        ("US VISA PHOTOS", [
            ("us_passport", "US visa photo (2x2 inch, white background)"),
            ("us_passport_sheet", "Print sheet (5x7 inch) with 4 photos"),
        ]),
        ("MALAYSIAN PASSPORT PHOTOS", [
            ("my_passport", "Malaysian passport (35x50mm, white background)"),
            ("my_passport_sheet", "Print sheet (4x6 inch) with 4 photos"),
        ]),
        ("UK PASSPORT PHOTOS", [
            ("uk_passport", "UK passport (35x45mm, white background)"),
            ("uk_passport_sheet", "Print sheet (4x6 inch) with 4 photos"),
        ]),
        ("SCHOOL ID PHOTOS", [
            ("school_id", "School ID (40x50mm, blue background)"),
            ("school_id_sheet", "Print sheet (4x6 inch) with 4 photos"),
        ]),
        ("JPJ LICENSE PHOTOS", [
            ("jpj_license", "JPJ license (25x32mm, white background, 40% face)"),
            ("jpj_license_sheet", "Print sheet (4x6 inch) with 4 photos"),
        ]),
    ]

    num = 1
    for category, items in categories:
        if category != "Base outputs":
            print(f"\n  {category}:")
        for key, desc in items:
            if key in outputs:
                print(f"  {num}. {outputs[key]}")
                print(f"     - {desc}")
                num += 1

    print("="*70)
