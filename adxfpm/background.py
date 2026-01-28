"""
Background removal and filling module
"""

import gc
import logging
import traceback
from enum import Enum
from typing import Tuple

from PIL import Image
from rembg import remove, new_session

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class BackgroundRemovalError(Exception):
    """Custom exception for background removal failures"""
    pass


class MemoryError(BackgroundRemovalError):
    """Memory-related errors (CUDA OOM, system memory)"""
    pass


class SessionError(BackgroundRemovalError):
    """Session initialization or corruption errors"""
    pass


# =============================================================================
# FILL STRATEGY
# =============================================================================

class FillStrategy(Enum):
    """Background fill strategy."""
    WHITE = (255, 255, 255)
    BLUE = (0, 127, 255)

    @staticmethod
    def from_color(color: Tuple[int, int, int]) -> 'FillStrategy':
        """Get strategy from RGB tuple, or create a custom one."""
        for s in FillStrategy:
            if s.value == color:
                return s
        # For custom colors, just return the color directly via CUSTOM handling
        return None

    @staticmethod
    def fill(img_with_alpha: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
        """Fill transparent areas with the given color."""
        bg = Image.new('RGB', img_with_alpha.size, color)
        bg.paste(img_with_alpha, (0, 0), img_with_alpha)
        return bg


# =============================================================================
# BACKGROUND REMOVER CLASS
# =============================================================================

class BackgroundRemover:
    """Manages background removal sessions and operations."""

    def __init__(self, force_cpu: bool = False):
        self._force_cpu = force_cpu
        self._session = None
        self._init_count = 0
        self._init_session()

    def _clear_gpu_memory(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleared")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to clear GPU memory: {e}")

    def _init_session(self) -> None:
        logger.info("Initializing rembg session...")
        self._init_count += 1

        if not self._force_cpu:
            try:
                self._session = new_session(
                    model_name="birefnet-general",
                    providers=[
                        ('CUDAExecutionProvider', {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        }),
                        'CPUExecutionProvider'
                    ]
                )
                logger.info("Rembg session initialized with CUDA (birefnet-general)")
                return
            except Exception as e:
                logger.warning(f"CUDA initialization failed: {e}")
                logger.info("Falling back to CPU...")

        try:
            self._session = new_session(model_name="birefnet-general")
            logger.info("Rembg session initialized with CPU (birefnet-general)")
        except Exception as e:
            logger.error(f"Failed to initialize rembg session: {e}")
            logger.error(traceback.format_exc())
            raise SessionError(f"Cannot initialize rembg session: {e}")

    @property
    def session(self):
        return self._session

    @property
    def init_count(self) -> int:
        return self._init_count

    def reinitialize(self, force_cpu: bool = False) -> None:
        """Re-create the session (e.g. after memory error)."""
        logger.warning("Reinitializing rembg session...")
        self._session = None
        self._force_cpu = force_cpu or (self._init_count >= 3)
        if self._force_cpu:
            logger.warning("Multiple failures detected, forcing CPU mode...")
        self._init_session()

    def remove(self, img: Image.Image, max_retries: int = 2) -> Image.Image:
        """Remove background from PIL Image with retry logic.

        Returns:
            PIL Image with transparent background.

        Raises:
            MemoryError: If memory issues persist after retries.
            BackgroundRemovalError: For other processing failures.
        """
        logger.info(f"Removing background (image size: {img.size})")
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                output = remove(
                    img,
                    session=self._session,
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10,
                )
                logger.info("Background removed successfully")
                return output

            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.error(f"Background removal failed (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                logger.debug(traceback.format_exc())

                if self._is_memory_error(e):
                    logger.warning("Memory error detected, clearing GPU memory...")
                    self._clear_gpu_memory()
                    if attempt < max_retries:
                        logger.info(f"Retrying after memory cleanup (attempt {attempt + 2})...")
                        continue
                    else:
                        raise MemoryError(f"Memory error after {max_retries + 1} attempts: {error_msg}")
                else:
                    raise BackgroundRemovalError(f"Background removal failed: {error_msg}")

        raise BackgroundRemovalError(f"Background removal failed after {max_retries + 1} attempts: {last_error}")

    def fill(self, img_with_alpha: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
        """Fill transparent areas with the given color."""
        return FillStrategy.fill(img_with_alpha, color)

    @staticmethod
    def _is_memory_error(error: Exception) -> bool:
        error_str = str(error).lower()
        memory_indicators = [
            'out of memory', 'oom', 'cuda error', 'cudnn',
            'memory allocation', 'alloc', 'cannot allocate',
            'memory limit', 'insufficient memory', 'ran out of memory',
            'no space left', 'memory exhausted',
        ]
        return any(ind in error_str for ind in memory_indicators)

    def close(self) -> None:
        """Release resources."""
        self._session = None
        self._clear_gpu_memory()


# =============================================================================
# Module-level convenience functions (backward compatibility)
# =============================================================================

def init_rembg_session(force_cpu=False):
    remover = BackgroundRemover(force_cpu=force_cpu)
    return remover.session


def remove_background(img, session, max_retries=2):
    logger.info(f"Removing background (image size: {img.size})")
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            output = remove(
                img, session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )
            logger.info("Background removed successfully")
            return output
        except Exception as e:
            last_error = e
            logger.error(f"Background removal failed (attempt {attempt + 1}): {e}")
            if BackgroundRemover._is_memory_error(e):
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                if attempt < max_retries:
                    continue
                raise MemoryError(f"Memory error after {max_retries + 1} attempts: {e}")
            else:
                raise BackgroundRemovalError(f"Background removal failed: {e}")
    raise BackgroundRemovalError(f"Background removal failed: {last_error}")


def fill_with_white(img_with_alpha):
    print("\nFilling transparency with white...")
    result = FillStrategy.fill(img_with_alpha, FillStrategy.WHITE.value)
    print("White background applied")
    return result


def fill_with_blue(img_with_alpha):
    print("\nFilling transparency with blue...")
    result = FillStrategy.fill(img_with_alpha, FillStrategy.BLUE.value)
    print("Blue background applied")
    return result
