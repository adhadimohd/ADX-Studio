# ADX Studio - ADX FPM (Face Photo Processor)

Passport and ID photo processing system with automatic face detection, background removal, format cropping, print sheet generation, and optional AI enhancement.

## Tech Stack

- **Python 3.x** with Flask web framework
- **PyTorch** + **ONNX Runtime GPU** for GPU-accelerated inference
- **MediaPipe** for face landmark detection
- **Rembg** for AI background removal
- **Pillow** / **OpenCV** for image processing
- **Replicate API** / local Flux pipeline for AI portrait enhancement

## Project Structure

```
adxfpm/                    # Main package
  config.py                # PhotoFormat registry, format specs, global settings
  processor.py             # PhotoProcessor facade — orchestrates full pipeline
  face_detection.py        # MediaPipe face landmark detection
  background.py            # Background removal (rembg) + fill strategies
  photo_formats.py         # Landmark-based cropping + adjustments
  sheet.py                 # 2x2 print sheet generation with cutting marks
  flux_enhance.py          # AI enhancement (Replicate API + local Flux)
  utils.py                 # GPU info utilities
  web.py                   # Flask routes (/api/process, /api/formats, etc.)
  main.py                  # CLI batch entry point
  templates/index.html     # Single-page web UI
run_web.py                 # Web server launcher
test_local.py              # Local integration test
```

## Running

```bash
# Web server (http://127.0.0.1:8080)
python run_web.py

# CLI batch processing (reads input/passport.jpg)
python adxfpm/main.py

# Test
python test_local.py
```

## Architecture

- **Registry pattern**: `PhotoFormatRegistry` in `config.py` holds all format specs as frozen `PhotoFormat` dataclasses (US, Malaysia, UK, School, JPJ, China, Japan, Umrah)
- **Facade pattern**: `PhotoProcessor` in `processor.py` orchestrates: face detection -> crop -> bg removal -> format -> sheet -> optional AI enhance
- **Pipeline**: `FaceDetector` -> `PhotoFormatter` -> `SheetGenerator` -> `PhotoEnhancer`
- Resource-heavy components (`BackgroundRemover`, `PhotoEnhancer`) have explicit `close()`/`reinitialize()` lifecycle methods

## Key Settings (config.py)

- `DPI = 300`, output to `outputs/adxfpm/`
- Each `PhotoFormat` defines: `size_mm/inches`, `sheet_inches`, `bg_color`, `face_coverage`, `nose_position`, `mark_length/offset`
- CUDA used when available, falls back to CPU

## Web API

- `POST /api/process` — upload photo with format key, optional adjustments (brightness, rotation, face_scale, nose_offset), optional AI enhance mode+prompt
- `GET /api/formats` — list available formats
- `GET /api/health` — health check with GPU info
- `GET /output/<id>/<file>` — serve generated images

## Conventions

- One responsibility per module
- Type hints throughout (dataclasses, Optional, Tuple, Dict)
- Centralized config with registry pattern — add new formats in `config.py` only
- Standard Python logging to `logs/adxfpm.log`
- Upload limit: 32MB
