"""
Download Spark-TTS model from Hugging Face
"""
from huggingface_hub import snapshot_download
import os

print("=" * 60)
print("Downloading Spark-TTS Model")
print("=" * 60)

model_dir = "Spark-TTS-0.5B"

if os.path.exists(model_dir):
    print(f"✓ Model already exists at {model_dir}")
    response = input("Re-download? (y/n): ")
    if response.lower() != 'y':
        print("Skipping download")
        exit(0)

print("\n📥 Downloading unsloth/Spark-TTS-0.5B...")
print("   This may take several minutes...")

snapshot_download("unsloth/Spark-TTS-0.5B", local_dir=model_dir)

print("\n✓ Model downloaded successfully!")
print(f"   Location: {model_dir}")
print("\n⏭️  Next step: Run 2_data_preparation.py")