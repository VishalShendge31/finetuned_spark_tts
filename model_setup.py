"""
Step 3: Model Setup and Initialization
Setup Spark-TTS model with LoRA adapters
"""
import sys
sys.path.append('Spark-TTS')

import torch
from unsloth import FastLanguageModel
import config
from utils import get_huggingface_token, print_gpu_stats

def setup_model(lora_r=None, lora_alpha=None):
    """
    Setup Spark-TTS model with LoRA adapters
    
    Args:
        lora_r: LoRA rank (defaults to config.LORA_R)
        lora_alpha: LoRA alpha (defaults to config.LORA_ALPHA)
    
    Returns:
        model, tokenizer
    """
    if lora_r is None:
        lora_r = config.LORA_R
    if lora_alpha is None:
        lora_alpha = config.LORA_ALPHA
    
    hf_token = get_huggingface_token()
    
    print(f"\n📦 Loading Spark-TTS base model...")
    print(f"   Model: {config.MODEL_NAME}")
    print(f"   Max sequence length: {config.MAX_SEQ_LENGTH}")
    print(f"   Data type: {config.DTYPE}")
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=config.LOAD_IN_4BIT,
        token=hf_token,
    )
    
    print("✓ Base model loaded")
    
    # Add LoRA adapters
    print(f"\n🔧 Adding LoRA adapters...")
    print(f"   Rank (r): {lora_r}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {config.LORA_DROPOUT}")
    print(f"   Target modules: {len(config.TARGET_MODULES)}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=config.TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✓ LoRA adapters added")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / total_params
    
    print(f"\n📊 Model parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({trainable_percent:.2f}%)")
    
    return model, tokenizer

def main():
    print("=" * 60)
    print("Step 3: Model Setup")
    print("=" * 60)
    
    # Setup model
    model, tokenizer = setup_model()
    
    print_gpu_stats()
    
    print("\n" + "=" * 60)
    print("✅ Model setup complete!")
    print("=" * 60)
    print("\n⏭️  Next step: Run 4_hyperparameter_tuning.py")

if __name__ == "__main__":
    main()