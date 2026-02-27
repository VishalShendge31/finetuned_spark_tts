"""
Configuration file for Spark-TTS fine-tuning
OPTIMIZED FOR: RTX 5070 12GB VRAM + German TTS
"""
import os
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["UNSLOTH_ATTENTION"] = "torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch

# ============= Paths =============

OUTPUT_DIR = "outputs"
BEST_MODEL_DIR = "best_model"
HYPERPARAMETER_RESULTS_DIR = "hyperparameter_results"
SPARK_TTS_MODEL_DIR = "Spark-TTS-0.5B"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(HYPERPARAMETER_RESULTS_DIR, exist_ok=True)

# ============= Hugging Face =============

HUGGINGFACE_TOKEN = "your_huggingface_token_here"

# ============= Model Configuration =============

MODEL_NAME = "Spark-TTS-0.5B/LLM"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.float32  # Spark-TTS requires float32
LOAD_IN_4BIT = False    # Not compatible with float32

# ============= Dataset Configuration =============

DATASET_NAME = "Vishalshendge3198/Dataset_eleven_v3"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 3407

# ============= Audio Processing =============

SAMPLE_RATE = 16000  # Spark-TTS uses 16kHz
TARGET_SAMPLE_RATE = 16000

# ============= LoRA Configuration =============

# Note: Spark-TTS with float32 base requires bfloat16 for LoRA

LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# ============= Training Configuration =============

TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 5,
    "learning_rate": 5e-4,
    "weight_decay": 0.001,
    "lr_scheduler_type": "linear",
    "optim": "adamw_8bit",
    "logging_steps": 1,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "seed": RANDOM_SEED,
    "fp16": False,   # Spark-TTS uses float32
    "bf16": False,   # Spark-TTS uses float32
    "gradient_checkpointing": True,
}

# ============= Hyperparameter Search Space =============
#trail 1:
'''HYPERPARAMETER_SEARCH_SPACE = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
    "gradient_accumulation_steps": [2, 4],
    "lora_r": [16, 32],
    "num_train_epochs": [2],
}'''

#trail 2:
'''HYPERPARAMETER_SEARCH_SPACE = {
    "learning_rate": [3e-4, 4e-4, 5e-4, 6e-4, 7e-4],
    "gradient_accumulation_steps": [2],
    "lora_r": [32, 48, 64],
    "num_train_epochs": [2, 3],
}'''

HYPERPARAMETER_SEARCH_SPACE = {
    "learning_rate": [4e-4, 5e-4, 6e-4],  
    "gradient_accumulation_steps": [2],     
    "lora_r": [64, 96, 128],                
    "num_train_epochs": [3, 4],             
}

FIXED_WARMUP_STEPS = 5
FIXED_WEIGHT_DECAY = 0.001
FIXED_BATCH_SIZE = 1

# ============= Inference Parameters =============

INFERENCE_PARAMS = {
    "max_new_tokens": 2048,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 1.0,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= GPU Stats Display =============

if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory = gpu_props.total_memory / 1024**3
    
    print("\n" + "="*70)
    print("Spark-TTS CONFIGURATION - RTX 5070 + GERMAN TTS")
    print("="*70)
    print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total      : {gpu_memory:.2f} GB")
    print(f"CUDA Version    : {torch.version.cuda}")
    
    print("\nMODEL SETTINGS:")
    print(f"  - Model         : Spark-TTS-0.5B")
    print(f"  - LoRA rank     : {LORA_R}")
    print(f"  - Batch size    : {TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  - Grad accum    : {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  - Learning rate : {TRAINING_CONFIG['learning_rate']}")
    
    total_combinations = 1
    for values in HYPERPARAMETER_SEARCH_SPACE.values():
        total_combinations *= len(values)
    
    print(f"\nGRID SEARCH: {total_combinations} combinations")
    print("="*70 + "\n")

print(f"Device: {DEVICE}")

