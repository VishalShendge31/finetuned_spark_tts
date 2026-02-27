"""
Setup environment for Spark-TTS fine-tuning
Optimized for RTX 5070 12GB + German TTS
"""
import subprocess
import sys
import os

# Disable XFormers for RTX 5070 compatibility

os.environ["XFORMERS_DISABLED"] = "1"
os.environ["UNSLOTH_ATTENTION"] = "torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

print("✅ Environment configured for RTX 5070 + Spark-TTS")
print("   • XFormers: Disabled")
print("   • Attention: PyTorch SDPA")

def check_virtual_environment():
    """Check if running in a virtual environment"""
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"In virtual environment: {in_venv}")
    
    if not in_venv:
        print("\n⚠ WARNING: Not running in a virtual environment!")
        print("\nTo activate tts_venv:")
        print("  PowerShell: .\\tts_venv\\Scripts\\Activate.ps1")
        print("  CMD: .\\tts_venv\\Scripts\\activate.bat")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please activate virtual environment first.")
            sys.exit(0)
    else:
        print("✓ Virtual environment is active")
    
    return in_venv

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_pytorch():
    """Install PyTorch with CUDA support for Windows"""
    print("\n" + "=" * 60)
    print("Installing PyTorch")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            
            response = input("\nPyTorch with CUDA already installed. Reinstall? (y/n): ")
            if response.lower() != 'y':
                return
        else:
            print("⚠ PyTorch installed but no CUDA support")
    except ImportError:
        print("PyTorch not found")
    
    print("\n🚀 Installing PyTorch with CUDA 12.1...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])
    
    # Verify installation
    import torch
    print(f"\n✓ PyTorch {torch.__version__} installed")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

def setup_environment():
    """Setup the environment and install required packages"""
    print("=" * 60)
    print("Spark-TTS Fine-tuning Environment Setup")
    print("Python 3.14 + tts_venv + Windows GPU Support")
    print("=" * 60)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Upgrade pip
    print("\n" + "=" * 60)
    print("Upgrading pip, wheel, and setuptools...")
    print("=" * 60)
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "--upgrade", "pip", "wheel", "setuptools"])
    
    # Install PyTorch
    install_pytorch()
    
    print("\n" + "=" * 60)
    print("Installing Core ML/DL Packages...")
    print("=" * 60)
    
    core_packages = [
        "transformers==4.56.2",
        "datasets>=3.4.1,<4.0.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl==0.22.2",
        "sentencepiece",
        "protobuf",
        "huggingface_hub>=0.34.0",
    ]
    
    for package in core_packages:
        try:
            print(f"\n📦 Installing {package}...")
            install_package(package)
            print(f"✓ {package} installed")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\n" + "=" * 60)
    print("Installing Audio Processing Packages...")
    print("=" * 60)
    
    audio_packages = [
        "torchaudio",
        "soundfile",
        "omegaconf",
        "einx",
    ]
    
    for package in audio_packages:
        try:
            print(f"\n📦 Installing {package}...")
            install_package(package)
            print(f"✓ {package} installed")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\n" + "=" * 60)
    print("Installing Utilities...")
    print("=" * 60)
    
    utility_packages = [
        "matplotlib",
        "seaborn",
        "pandas",
        "numpy",
        "scikit-learn",
        "ipython",
        "tqdm",
        "num2words",  # For German number normalization
    ]
    
    for package in utility_packages:
        try:
            print(f"\n📦 Installing {package}...")
            install_package(package)
            print(f"✓ {package} installed")
        except Exception as e:
            print(f"✗ Failed to install {package}: {e}")
    
    # Install unsloth
    print("\n" + "=" * 60)
    print("Installing Unsloth...")
    print("=" * 60)
    
    try:
        print("Attempting to install unsloth...")
        install_package("unsloth")
        print("✓ unsloth installed successfully")
    except Exception as e:
        print(f"⚠ Warning: Failed to install unsloth: {e}")
        print("\nVisit: https://github.com/unslothai/unsloth")
    
    # Clone Spark-TTS repository
    print("\n" + "=" * 60)
    print("Cloning Spark-TTS repository...")
    print("=" * 60)
    
    if not os.path.exists("Spark-TTS"):
        try:
            subprocess.check_call(["git", "clone", "https://github.com/SparkAudio/Spark-TTS"])
            print("✓ Spark-TTS repository cloned")
        except Exception as e:
            print(f"✗ Failed to clone Spark-TTS: {e}")
            print("Please manually clone: git clone https://github.com/SparkAudio/Spark-TTS")
    else:
        print("✓ Spark-TTS repository already exists")
    
    # Verify installations
    print("\n" + "=" * 60)
    print("Verifying Installations...")
    print("=" * 60)
    
    verifications = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("soundfile", "SoundFile"),
    ]
    
    all_good = True
    for module, name in verifications:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - FAILED")
            all_good = False
    
    # Special check for PyTorch CUDA
    try:
        import torch
        print(f"\n📊 System Info:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU memory: {total_memory:.2f} GB")
    except Exception as e:
        print(f"✗ PyTorch verification failed: {e}")
        all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("✓ Setup Complete - All packages installed successfully!")
    else:
        print("⚠ Setup Complete - Some packages failed (see above)")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Download Spark-TTS model:")
    print("   Run 1_download_model.py")
    print("\n2. Prepare dataset:")
    print("   Run 2_data_preparation.py")
    print("\n3. Run the pipeline:")
    print("   python 3_model_setup.py")
    print("   python 4_hyperparameter_tuning.py")
    print("   python 5_train_full.py")
    print("   python 6_inference.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    setup_environment()