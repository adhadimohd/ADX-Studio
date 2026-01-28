"""
Flux Klein AI Enhancement Module for passport photos

Supports two modes:
1. Replicate API (fast, cloud-based) - Primary
2. Local pipeline (slower, GPU required) - Fallback
"""

import os
import gc
import io
import base64
import logging
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import requests
import torch
from PIL import Image, ImageDraw, ImageFont

from . import config

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class FluxEnhanceError(Exception):
    """Custom exception for Flux enhancement failures"""
    pass


class FluxMemoryError(FluxEnhanceError):
    """Memory-related errors during Flux processing"""
    pass


class FluxAPIError(FluxEnhanceError):
    """API-related errors during Flux processing"""
    pass


# =============================================================================
# PROMPTS
# =============================================================================

ENHANCE_PROMPTS = {
    'retouch': {
        'prefix': "Professional portrait retouching: ",
        'suffix': " Maintain exact facial features, natural skin texture, subtle enhancement. Professional headshot quality, even lighting, sharp focus."
    }
}


# =============================================================================
# PHOTO ENHANCER CLASS
# =============================================================================

class PhotoEnhancer:
    """AI photo enhancement via Replicate API or local Flux pipeline."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    REPLICATE_OUTPUT_DIR = PROJECT_ROOT / "output" / "replicate"

    def __init__(self, mode: str = config.FLUX_MODE):
        self._mode = mode
        self._api_token = config.REPLICATE_API_TOKEN
        self._model_version = config.REPLICATE_MODEL_VERSION
        self._api_url = config.REPLICATE_API_URL
        self._flux_pipeline = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def get_processing_mode(self) -> str:
        if self._mode == 'local':
            return 'local'
        elif self._mode == 'api':
            if self._api_token:
                return 'api'
            logger.warning("FLUX_MODE=api but REPLICATE_API_TOKEN not set, falling back to local")
            return 'local'
        else:  # auto
            return 'api' if self._api_token else 'local'

    def get_info(self) -> Dict:
        mode = self.get_processing_mode()
        return {
            'name': 'FLUX.2-klein-4B',
            'mode': 'Replicate API' if mode == 'api' else 'Local GPU',
            'mode_short': 'API' if mode == 'api' else 'Local',
        }

    def enhance(
        self,
        image: Image.Image,
        prompt: str,
        mode: str = 'retouch',
        num_inference_steps: int = 4,
        guidance_scale: float = 1,
        output_size: Optional[Tuple[int, int]] = None,
        force_local: bool = False,
    ) -> Image.Image:
        """Enhance photo using Flux Klein (API or local).

        Returns:
            Enhanced PIL Image.

        Raises:
            FluxMemoryError, FluxAPIError, FluxEnhanceError
        """
        processing_mode = 'local' if force_local else self.get_processing_mode()
        logger.info(f"Starting Flux enhancement (mode: {mode}, processing: {processing_mode})")

        image = self._resize_to_max_edge(image, max_edge=1920)
        full_prompt = self._build_prompt(mode, prompt)

        if output_size:
            width, height = output_size
        else:
            width, height = image.size
        width = (width // 8) * 8
        height = (height // 8) * 8

        if processing_mode == 'api':
            try:
                if image.size != (width, height):
                    api_image = image.resize((width, height), Image.LANCZOS)
                else:
                    api_image = image

                padded_image, original_size, padding_info = self._pad_to_square(api_image)
                result = self._call_replicate_api(full_prompt, padded_image, original_size, padding_info)
                logger.info("Flux enhancement via API completed successfully")

                if result.size != (width, height):
                    result = result.resize((width, height), Image.LANCZOS)
                return result

            except FluxAPIError as e:
                logger.warning(f"API processing failed: {e}")
                logger.info("Falling back to local processing...")

        return self._enhance_local(image, full_prompt, width, height, num_inference_steps, guidance_scale)

    def release(self) -> None:
        if self._flux_pipeline is not None:
            logger.info("Releasing Flux pipeline...")
            del self._flux_pipeline
            self._flux_pipeline = None
            self._clear_gpu_memory()
            logger.info("Flux pipeline released")

    # -----------------------------------------------------------------
    # Private — API
    # -----------------------------------------------------------------

    def _call_replicate_api(self, prompt, image, original_size=None, padding_info=None, output_format="jpg"):
        if not self._api_token:
            raise FluxAPIError("REPLICATE_API_TOKEN not set")

        input_width, input_height = image.size
        logger.info(f"Calling Replicate API (input: {input_width}x{input_height})...")

        image_uri = self._image_to_data_uri(image)

        headers = {
            "Authorization": f"Token {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "version": self._model_version,
            "input": {
                "prompt": prompt,
                "images": [image_uri],
                "aspect_ratio": "1:1",
                "output_format": output_format,
                "output_quality": 95,
                "output_megapixels": "4",
                "go_fast": True,
            },
        }

        try:
            response = requests.post(self._api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            prediction = response.json()

            prediction_id = prediction.get('id')
            prediction_url = prediction.get('urls', {}).get('get') or f"{self._api_url}/{prediction_id}"
            logger.info(f"Prediction created: {prediction_id}")

            max_wait, poll_interval, elapsed = 300, 2, 0
            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                status_response = requests.get(prediction_url, headers=headers, timeout=30)
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get('status')

                if status == 'succeeded':
                    output = status_data.get('output')
                    if output and len(output) > 0:
                        img_response = requests.get(output[0], timeout=60)
                        img_response.raise_for_status()
                        result_image = Image.open(io.BytesIO(img_response.content))

                        if result_image.size != (input_width, input_height):
                            result_image = result_image.resize((input_width, input_height), Image.LANCZOS)

                        try:
                            self._save_comparison_image(image, result_image, prompt)
                        except Exception as e:
                            logger.warning(f"Failed to save comparison: {e}")

                        if original_size and padding_info:
                            result_image = self._crop_from_square(result_image, original_size, padding_info)

                        return result_image
                    raise FluxAPIError("API returned success but no output images")

                elif status == 'failed':
                    raise FluxAPIError(f"Prediction failed: {status_data.get('error', 'Unknown')}")
                elif status == 'canceled':
                    raise FluxAPIError("Prediction was canceled")

            raise FluxAPIError(f"Prediction timed out after {max_wait}s")

        except requests.exceptions.RequestException as e:
            raise FluxAPIError(f"API request failed: {e}")

    # -----------------------------------------------------------------
    # Private — Local
    # -----------------------------------------------------------------

    def _init_flux_pipeline(self):
        if self._flux_pipeline is not None:
            return self._flux_pipeline

        logger.info("Initializing Flux Klein pipeline (local)...")
        try:
            from diffusers import Flux2KleinPipeline

            repo_id = "black-forest-labs/FLUX.2-klein-4B"
            self._flux_pipeline = Flux2KleinPipeline.from_pretrained(
                repo_id, torch_dtype=torch.bfloat16
            )
            if torch.cuda.is_available():
                self._flux_pipeline.enable_model_cpu_offload()
                logger.info("Flux Klein initialized with CUDA + CPU offload")
            else:
                logger.warning("CUDA not available, running on CPU (slow)")

            return self._flux_pipeline
        except Exception as e:
            logger.error(f"Failed to initialize Flux Klein: {e}")
            raise FluxEnhanceError(f"Cannot initialize Flux Klein: {e}")

    def _enhance_local(self, image, full_prompt, width, height, num_inference_steps, guidance_scale):
        pipeline = self._init_flux_pipeline()

        if image.size != (width, height):
            input_image = image.resize((width, height), Image.LANCZOS)
        else:
            input_image = image

        logger.info(f"Processing locally: {width}x{height}, steps: {num_inference_steps}")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            generator = torch.Generator(device=device)
            result = pipeline(
                prompt=full_prompt, image=input_image,
                height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, generator=generator,
            ).images[0]
            logger.info("Local enhancement completed")
            return result
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ['out of memory', 'oom', 'cuda error', 'alloc']):
                self._clear_gpu_memory()
                raise FluxMemoryError(f"GPU memory error: {e}")
            raise FluxEnhanceError(f"Enhancement failed: {e}")

    # -----------------------------------------------------------------
    # Private — helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _build_prompt(mode: str, user_prompt: str) -> str:
        if mode not in ENHANCE_PROMPTS:
            return user_prompt
        cfg = ENHANCE_PROMPTS[mode]
        return cfg['prefix'] + user_prompt + cfg['suffix']

    @staticmethod
    def _clear_gpu_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _resize_to_max_edge(image: Image.Image, max_edge: int = 1920) -> Image.Image:
        w, h = image.size
        longest = max(w, h)
        if longest <= max_edge:
            return image
        scale = max_edge / longest
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    @staticmethod
    def _image_to_data_uri(image: Image.Image) -> str:
        buffer = io.BytesIO()
        img = image.convert('RGB') if image.mode in ('RGBA', 'LA', 'P') else image
        img.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"

    @staticmethod
    def _pad_to_square(image):
        width, height = image.size
        original_size = (width, height)
        if width == height:
            return image, original_size, {'padded': False}

        target_size = max(width, height)
        img_rgb = image.convert('RGB')

        pixels = []
        for x in range(width):
            pixels.append(img_rgb.getpixel((x, 0)))
            pixels.append(img_rgb.getpixel((x, height - 1)))
        for y in range(height):
            pixels.append(img_rgb.getpixel((0, y)))
            pixels.append(img_rgb.getpixel((width - 1, y)))

        avg_r = sum(p[0] for p in pixels) // len(pixels)
        avg_g = sum(p[1] for p in pixels) // len(pixels)
        avg_b = sum(p[2] for p in pixels) // len(pixels)

        square_img = Image.new('RGB', (target_size, target_size), (avg_r, avg_g, avg_b))
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        square_img.paste(img_rgb, (paste_x, paste_y))

        return square_img, original_size, {
            'padded': True, 'original_width': width, 'original_height': height,
            'target_size': target_size, 'paste_x': paste_x, 'paste_y': paste_y,
        }

    @staticmethod
    def _crop_from_square(image, original_size, padding_info):
        if not padding_info.get('padded', False):
            return image
        out_w, out_h = image.size
        orig_w, orig_h = original_size
        orig_ratio = orig_w / orig_h

        if orig_ratio > 1:
            crop_w, crop_h = out_w, int(out_w / orig_ratio)
        else:
            crop_h, crop_w = out_h, int(out_h * orig_ratio)

        left = (out_w - crop_w) // 2
        top = (out_h - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))

    def _save_comparison_image(self, input_image, output_image, prompt=""):
        self.REPLICATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        filepath = self.REPLICATE_OUTPUT_DIR / f"compare_{timestamp}.jpg"

        in_w, in_h = input_image.size
        out_w, out_h = output_image.size

        label_height, gap = 40, 20
        max_height = max(in_h, out_h)
        canvas = Image.new('RGB', (in_w + gap + out_w, max_height + label_height), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        canvas.paste(input_image.convert('RGB'), (0, label_height + (max_height - in_h) // 2))
        canvas.paste(output_image.convert('RGB'), (in_w + gap, label_height + (max_height - out_h) // 2))

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), f"INPUT: {in_w}x{in_h}", fill=(100, 200, 255), font=font)
        draw.text((in_w + gap + 10, 10), f"OUTPUT: {out_w}x{out_h}", fill=(100, 255, 100), font=font)

        canvas.save(filepath, 'JPEG', quality=95)
        logger.info(f"Comparison saved: {filepath}")
        return filepath


# =============================================================================
# Module-level convenience functions (backward compatibility)
# =============================================================================

_default_enhancer = None


def _get_default_enhancer() -> PhotoEnhancer:
    global _default_enhancer
    if _default_enhancer is None:
        _default_enhancer = PhotoEnhancer()
    return _default_enhancer


def get_processing_mode():
    return _get_default_enhancer().get_processing_mode()


def get_pipeline_info():
    return _get_default_enhancer().get_info()


def build_prompt(mode, user_prompt):
    return PhotoEnhancer._build_prompt(mode, user_prompt)


def enhance_photo(image, prompt, mode='retouch', num_inference_steps=4,
                  guidance_scale=1, output_size=None, force_local=False):
    return _get_default_enhancer().enhance(
        image, prompt, mode=mode,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        output_size=output_size,
        force_local=force_local,
    )


def init_flux_pipeline():
    return _get_default_enhancer()._init_flux_pipeline()


def get_flux_pipeline():
    return _get_default_enhancer()._init_flux_pipeline()


def release_flux_pipeline():
    _get_default_enhancer().release()
