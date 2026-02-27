"""
Check if all required packages are installed
"""
import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - MISSING")
        return False

print("=" * 60)
print("Dependency Check for Spark-TTS")
print("=" * 60)

packages = [
    ("PyTorch", "torch"),
    ("Transformers", "transformers"),
    ("Datasets", "datasets"),
    ("Accelerate", "accelerate"),
    ("PEFT", "peft"),
    ("TRL", "trl"),
    ("Bitsandbytes", "bitsandbytes"),
    ("Hugging Face Hub", "huggingface_hub"),
    ("SentencePiece", "sentencepiece"),
    ("Protobuf", "google.protobuf"),
    ("Torchaudio", "torchaudio"),
    ("SoundFile", "soundfile"),
    ("OmegaConf", "omegaconf"),
    ("Einx", "einx"),
    ("Matplotlib", "matplotlib"),
    ("Seaborn", "seaborn"),
    ("Pandas", "pandas"),
    ("NumPy", "numpy"),
    ("Scikit-learn", "sklearn"),
    ("Tqdm", "tqdm"),
    ("Num2words", "num2words"),
]

print("\n📦 Checking packages...\n")

all_installed = True
missing = []

for name, import_name in packages:
    if not check_package(name, import_name):
        all_installed = False
        missing.append(name)

print("\n" + "=" * 60)

if all_installed:
    print("✅ All packages installed!")
else:
    print(f"❌ Missing {len(missing)} package(s):")
    for pkg in missing:
        print(f"   - {pkg}")
    print("\nInstall missing packages:")
    print("pip install omegaconf einx num2words")

# Check PyTorch CUDA

print("\n" + "=" * 60)
print("PyTorch Configuration")
print("=" * 60)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ Warning: CUDA not available")
except Exception as e:
    print(f"Error checking PyTorch: {e}")

print("=" * 60)