#!/usr/bin/env python3
"""
ADX FPM - Web Interface Runner
Run this script to start the web server
"""

import warnings
import os

# Suppress warnings before imports
os.environ['ORT_LOGGING_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from adxfpm.web import run_server

if __name__ == "__main__":
    run_server(host='127.0.0.1', port=8080, debug=True)
