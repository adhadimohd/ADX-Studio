#!/usr/bin/env python3
"""
ADX FPM - Web Interface
Flask-based web app for passport photo processing
"""

import warnings
import os
import uuid
import logging
import traceback
from pathlib import Path

os.environ['ORT_LOGGING_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch

from .config import DPI, REGISTRY, PhotoFormat, get_format_list
from .processor import PhotoProcessor
from .flux_enhance import ENHANCE_PROMPTS, FluxEnhanceError, FluxMemoryError, FluxAPIError
from .utils import GPUInfo

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
UPLOAD_FOLDER = PROJECT_ROOT / 'uploads'
OUTPUT_BASE = PROJECT_ROOT / 'outputs' / 'adxfpm'
LOG_DIR = PROJECT_ROOT / 'logs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'adxfpm.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# =============================================================================
# PHOTO PROCESSOR (single instance)
# =============================================================================

_processor: PhotoProcessor = None


def _get_processor() -> PhotoProcessor:
    global _processor
    if _processor is None:
        logger.info("Creating PhotoProcessor instance...")
        _processor = PhotoProcessor()
    return _processor


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large. Maximum file size is 32MB.',
        'error_type': 'file_too_large',
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'error_type': 'server_error',
        'error_details': str(error),
    }), 500


# =============================================================================
# HELPERS
# =============================================================================

def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    # Build a formats dict compatible with the existing template
    formats = {}
    for key, fmt in REGISTRY:
        formats[key] = {
            'name': fmt.name,
            'size': fmt.description,
            'description': fmt.name,
            'bg_color': 'blue' if fmt.bg_color == (0, 127, 255) else 'white',
            'sheet_inches': f"{int(fmt.sheet_inches[0])}x{int(fmt.sheet_inches[1])}",
        }
    return render_template('index.html', formats=formats)


@app.route('/api/formats')
def get_formats():
    formats = {}
    for key, fmt in REGISTRY:
        formats[key] = {
            'name': fmt.name,
            'size': fmt.description,
            'description': fmt.name,
            'bg_color': 'blue' if fmt.bg_color == (0, 127, 255) else 'white',
        }
    return jsonify(formats)


@app.route('/api/enhance-modes')
def get_enhance_modes():
    modes = {
        'none': {'name': 'None', 'description': 'No AI enhancement', 'placeholder': ''},
        'retouch': {
            'name': 'Retouch',
            'description': 'Professional portrait retouching',
            'placeholder': 'smooth skin, remove blemishes, brighten eyes, natural look',
            'prefix': ENHANCE_PROMPTS['retouch']['prefix'],
            'suffix': ENHANCE_PROMPTS['retouch']['suffix'],
        },
    }
    return jsonify(modes)


@app.route('/api/process', methods=['POST'])
def process():
    logger.info("=== New photo processing request ===")

    # Validate file
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'error_type': 'validation_error'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected', 'error_type': 'validation_error'}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, WEBP',
            'error_type': 'validation_error',
        }), 400

    # Validate format
    format_key = request.form.get('format', 'malaysia')
    if format_key not in REGISTRY:
        return jsonify({
            'error': 'Invalid photo format',
            'error_type': 'validation_error',
            'error_details': f"Available: {', '.join(REGISTRY.keys())}",
        }), 400

    # Fine adjustments
    adjustments = {
        'brightness': max(-100, min(100, float(request.form.get('brightness', 0)))),
        'rotation': max(-15, min(15, float(request.form.get('rotation', 0)))),
        'face_scale': max(-20, min(20, float(request.form.get('face_scale', 0)))),
        'nose_offset': max(-20, min(20, float(request.form.get('nose_offset', 0)))),
    }

    # Enhancement options
    enhance_mode = request.form.get('enhance_mode', '') or None
    enhance_prompt = request.form.get('enhance_prompt', '') or None
    if enhance_mode and enhance_mode not in ('retouch', 'change_cloth'):
        enhance_mode = None
    if enhance_mode and not (enhance_prompt and enhance_prompt.strip()):
        enhance_mode = None
        enhance_prompt = None

    # Save upload
    output_id = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    input_path = os.path.join(str(UPLOAD_FOLDER), f'{output_id}_{filename}')

    try:
        file.save(input_path)
        logger.info(f"File saved: {input_path} (format: {format_key})")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return jsonify({'error': 'Failed to save file', 'error_type': type(e).__name__}), 500

    try:
        processor = _get_processor()
        fmt = REGISTRY.get(format_key)

        # Output paths
        output_dir = OUTPUT_BASE / output_id
        os.makedirs(output_dir, exist_ok=True)

        result = processor.process(
            input_path, format_key,
            enhance_mode=enhance_mode, enhance_prompt=enhance_prompt,
            adjustments=adjustments,
        )

        # Save outputs
        photo_path = str(output_dir / f'{format_key}_photo.jpg')
        original_path = str(output_dir / f'{format_key}_original.jpg')
        sheet_path = str(output_dir / f'{format_key}_sheet.jpg')

        result.photo.save(photo_path, 'JPEG', quality=95, dpi=(DPI, DPI))
        result.original_photo.save(original_path, 'JPEG', quality=95, dpi=(DPI, DPI))
        result.sheet.save(sheet_path, 'JPEG', quality=95, dpi=(DPI, DPI))

        response_data = {
            'success': True,
            'output_id': output_id,
            'format': fmt.name,
            'size': fmt.description,
            'original_size': f"{result.original_photo.size[0]}x{result.original_photo.size[1]}",
            'photo_url': f'/output/{output_id}/{format_key}_photo.jpg',
            'original_url': f'/output/{output_id}/{format_key}_original.jpg',
            'sheet_url': f'/output/{output_id}/{format_key}_sheet.jpg',
        }

        if result.ai_photo:
            ai_photo_path = str(output_dir / f'{format_key}_ai_photo.jpg')
            result.ai_photo.save(ai_photo_path, 'JPEG', quality=95, dpi=(DPI, DPI))
            response_data['ai_photo_url'] = f'/output/{output_id}/{format_key}_ai_photo.jpg'

        if result.ai_sheet:
            ai_sheet_path = str(output_dir / f'{format_key}_ai_sheet.jpg')
            result.ai_sheet.save(ai_sheet_path, 'JPEG', quality=95, dpi=(DPI, DPI))
            response_data['ai_sheet_url'] = f'/output/{output_id}/{format_key}_ai_sheet.jpg'

        logger.info(f"Processing successful: {output_id}")
        return jsonify(response_data)

    except RuntimeError as e:
        logger.warning(f"Processing failed: {e}")
        return jsonify({'error': str(e), 'error_type': 'processing_error'}), 400

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'error_type': type(e).__name__,
            'error_details': traceback.format_exc(),
        }), 500

    finally:
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass


@app.route('/output/<output_id>/<filename>')
def serve_output(output_id, filename):
    file_path = OUTPUT_BASE / output_id / filename
    if file_path.exists():
        return send_file(str(file_path), mimetype='image/jpeg')
    return jsonify({'error': 'File not found'}), 404


@app.route('/download/<output_id>/<filename>')
def download_output(output_id, filename):
    file_path = OUTPUT_BASE / output_id / filename
    if file_path.exists():
        return send_file(str(file_path), as_attachment=True, download_name=filename)
    return jsonify({'error': 'File not found'}), 404


@app.route('/api/health')
def health_check():
    processor = _get_processor()
    return jsonify({
        'status': 'ok',
        'session_initialized': processor.bg_remover.session is not None,
        'session_reinit_count': processor.bg_remover.init_count,
    })


@app.route('/api/reinit', methods=['POST'])
def force_reinit():
    logger.info("Manual session reinitialization requested")
    try:
        processor = _get_processor()
        processor.reinitialize_bg()
        return jsonify({
            'success': True,
            'message': 'Session reinitialized',
            'reinit_count': processor.bg_remover.init_count,
        })
    except Exception as e:
        logger.error(f"Reinitialization failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/gpu-info')
def gpu_info():
    info = GPUInfo.get_info()
    enhancer_info = _get_processor().enhancer.get_info()

    return jsonify({
        'cuda_available': info.get('available', False),
        'device': info.get('device', 'CPU'),
        'gpu_name': info.get('name'),
        'gpu_memory_total': f"{info['vram_total_gb']} GB" if info.get('vram_total_gb') else None,
        'gpu_memory_free': f"{info['vram_free_gb']} GB" if info.get('vram_free_gb') else None,
        'flux_pipeline': enhancer_info['name'],
        'flux_mode': enhancer_info['mode'],
        'flux_mode_short': enhancer_info['mode_short'],
    })


# =============================================================================
# SERVER
# =============================================================================

def run_server(host='127.0.0.1', port=8080, debug=False):
    logger.info("=" * 70)
    logger.info("ADX FPM - Passport Photo Processor Web Interface")
    logger.info("=" * 70)
    GPUInfo.print_info()
    logger.info(f"Log file: {LOG_DIR / 'adxfpm.log'}")
    logger.info(f"Starting server at http://{host}:{port}")
    logger.info("=" * 70)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
