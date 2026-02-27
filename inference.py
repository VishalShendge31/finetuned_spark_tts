"""
Step 6: Inference with trained Spark-TTS model
"""
import sys
sys.path.append('Spark-TTS')

import torch
import numpy as np
from unsloth import FastLanguageModel
import config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
import os
import re

def generate_speech_from_text(
    model,
    tokenizer,
    audio_tokenizer,
    text: str,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_audio_tokens: int = 2048,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> np.ndarray:
    """Generate speech from text using Spark-TTS"""
    
    # Prepare model for inference
    FastLanguageModel.for_inference(model)
    
    # Create prompt
    prompt = "".join([
        "<|task_tts|>",
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_global_token|>"
    ])
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    
    print("Generating tokens...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_audio_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Trim input
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    
    # Decode
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    
    # Extract tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        print("Warning: No semantic tokens found")
        return np.array([], dtype=np.float32)
    
    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
    if not global_matches:
        print("Warning: No global tokens found")
        # Initialize with zeros if no tokens found (fallback)
        pred_global_ids = torch.zeros((1, 1, 32), dtype=torch.long)
    else:
        # Convert found tokens
        current_global_ids = [int(token) for token in global_matches]
        
        # ✅ CRITICAL FIX: Ensure exactly 32 tokens for the linear layer
        # Spark-TTS expects (Batch, 1, 32) global tokens
        if len(current_global_ids) < 32:
            print(f"Warning: Only {len(current_global_ids)} global tokens found. Padding to 32.")
            current_global_ids = current_global_ids + [0] * (32 - len(current_global_ids))
        elif len(current_global_ids) > 32:
            print(f"Warning: {len(current_global_ids)} global tokens found. Truncating to 32.")
            current_global_ids = current_global_ids[:32]
            
        pred_global_ids = torch.tensor(current_global_ids).long().unsqueeze(0).unsqueeze(0)
    
    print(f"Found {pred_semantic_ids.shape[1]} semantic tokens, {pred_global_ids.shape[2]} global tokens")
    
    # Detokenize
    print("Detokenizing...")
    audio_tokenizer.device = device
    audio_tokenizer.model.to(device)
    
    wav_np = audio_tokenizer.detokenize(
        pred_global_ids.to(device).squeeze(0),
        pred_semantic_ids.to(device)
    )
    
    return wav_np

def main():
    print("=" * 60)
    print("Step 6: Inference (Spark-TTS)")
    print("=" * 60)
    
    # Load model
    print("\nLoading trained model...")
    model_path = config.BEST_MODEL_DIR
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please run training first (5_train_full.py)")
        return
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    
    print("✓ Model loaded")
    
    # Load BiCodec tokenizer
    print("\nLoading BiCodec tokenizer...")
    audio_tokenizer = BiCodecTokenizer(config.SPARK_TTS_MODEL_DIR, "cuda")
    print("✓ BiCodec tokenizer loaded")
    
    # German prompts
    prompts = [
        "In der Ruhe liegt die Kraft.",
        "[happy] Das ist ja wunderbar! [laughing]",
        "[sighs] Ich bin so müde... [yawn]",
        "[angry] Das ist absolut inakzeptabel! [growl]",
        "[snarling] bleib mir bloss weg damit breathes out"
    ]
    
    # Create output directory
    output_dir = "output_audio"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Generating Audio")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt}")
        
        try:
            # Generate
            waveform = generate_speech_from_text(
                model, tokenizer, audio_tokenizer, prompt
            )
            
            if waveform.size > 0:
                # Save
                output_path = os.path.join(output_dir, f"output_{i+1}.wav")
                sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
                sf.write(output_path, waveform, sample_rate)
                print(f"✓ Saved: {output_path}")
                
                # Try to play
                try:
                    from IPython.display import Audio, display
                    display(Audio(waveform, rate=sample_rate))
                except:
                    pass
            else:
                print("✗ Generation failed")
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print(f"   Audio files saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()