"""
ADX FPM - Post-install bootstrap script.
Creates a virtual environment, installs dependencies, and sets up runtime directories.
Requires Python 3.10+ to be installed on the system.
"""

import subprocess
import sys
import os
import shutil

APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(APP_DIR, '.venv')
REQUIREMENTS = os.path.join(APP_DIR, 'requirements.txt')
RUNTIME_DIRS = ['input', 'outputs', 'uploads', 'logs']


def main():
    print("=" * 60)
    print("ADX FPM - Installer")
    print("=" * 60)

    # 1. Check Python version
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required, found {sys.version}")
        sys.exit(1)
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # 2. Create virtual environment
    if os.path.exists(VENV_DIR):
        print(f"[OK] Virtual environment already exists at {VENV_DIR}")
    else:
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
        print(f"[OK] Virtual environment created at {VENV_DIR}")

    # 3. Determine pip path
    if os.name == 'nt':
        pip_exe = os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
    else:
        pip_exe = os.path.join(VENV_DIR, 'bin', 'pip')

    # 4. Upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([pip_exe, 'install', '--upgrade', 'pip', '-q'])

    # 5. Install dependencies
    print("Installing dependencies (this may take several minutes)...")
    subprocess.check_call([pip_exe, 'install', '-r', REQUIREMENTS])
    print("[OK] All dependencies installed")

    # 6. Create runtime directories
    for d in RUNTIME_DIRS:
        path = os.path.join(APP_DIR, d)
        os.makedirs(path, exist_ok=True)
    print("[OK] Runtime directories created")

    # 7. Create .env from example if not exists
    env_file = os.path.join(APP_DIR, '.env')
    env_example = os.path.join(APP_DIR, '.env.example')
    if not os.path.exists(env_file) and os.path.exists(env_example):
        shutil.copy2(env_example, env_file)
        print("[OK] .env file created from .env.example")
    else:
        print("[OK] .env file already exists")

    print()
    print("=" * 60)
    print("Installation complete!")
    print("Run the app with: launcher.bat")
    print("Or manually: .venv\\Scripts\\activate && python run_web.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
