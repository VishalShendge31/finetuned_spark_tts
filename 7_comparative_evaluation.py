import os
import sys

# Set encoding for Windows terminal
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add working directory to path for relative imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'Spark-TTS'))

import torch
import json
import numpy as np
from datasets import load_from_disk
from unsloth import FastLanguageModel
from sparktts.models.audio_tokenizer import BiCodecTokenizer
import soundfile as sf
import re
import config
from tqdm import tqdm

# Paths and configuration
BASE_MODEL_NAME = config.MODEL_NAME
FINETUNED_MODEL_PATH = config.BEST_MODEL_DIR
EVAL_DIR = os.path.join(config.OUTPUT_DIR, "evaluation_comparison")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(os.path.join(EVAL_DIR, "audio_base"), exist_ok=True)
os.makedirs(os.path.join(EVAL_DIR, "audio_finetuned"), exist_ok=True)

def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """Tokenize dataset for evaluation"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

def evaluate_manual_loss(model, test_dataset):
    """Compute loss manually using forward pass"""
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            item = test_dataset[i]
            input_ids = torch.tensor(item["input_ids"]).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(item["attention_mask"]).unsqueeze(0).to(model.device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            count += 1
            
    return total_loss / count if count > 0 else 0

def generate_sample(model, tokenizer, audio_tokenizer, text, output_path):
    """Generate audio sample from prompt"""
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
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Generate
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Trim input
    generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
    
    # Decode to text to extract tokens
    predicts_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    
    # Extract semantic tokens
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
    if not semantic_matches:
        return False
    pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0).to(model.device)
    
    # Extract global tokens - Spark-TTS usually expects 32 global tokens
    global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
    if not global_matches:
        # Fallback to zeros if not found
        pred_global_ids = torch.zeros((1, 1, 32), dtype=torch.long).to(model.device)
    else:
        # Convert all found global tokens
        global_ids = [int(token) for token in global_matches]
        # Pad or truncate to match expected token_num (32)
        if len(global_ids) < 32:
            global_ids = global_ids + [0] * (32 - len(global_ids))
        else:
            global_ids = global_ids[:32]
        pred_global_ids = torch.tensor(global_ids).long().unsqueeze(0).unsqueeze(0).to(model.device)
    
    # Detokenize
    audio_tokenizer.device = model.device
    audio_tokenizer.model.to(model.device)
    
    waveform = audio_tokenizer.detokenize(
        pred_global_ids.squeeze(0),
        pred_semantic_ids
    )
    
    if waveform is not None and waveform.size > 0:
        sample_rate = audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(output_path, waveform, sample_rate)
        return True
    return False

def main():
    print("=" * 70)
    print("Comparative Evaluation: Base Model vs. Fine-tuned German Model")
    print("=" * 70)

    # 1. Load test dataset
    print("\nLoading test dataset...")
    try:
        dataset_dict = load_from_disk("processed_dataset")
        test_dataset_raw = dataset_dict["test"]
        print(f"Test samples: {len(test_dataset_raw)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Comparative Prompts
    prompts = [
        "In der Ruhe liegt die Kraft.",
        "[happy] Das ist ja wunderbar! [laughing]",
        "[sighs] Ich bin so müde... [yawn]",
        "[angry] Das ist absolut inakzeptabel! [growl]",
        "[snarling] bleib mir bloss weg damit breathes out"
    ]

    results = {"quantitative": {}, "qualitative_prompts": prompts}

    # 3. Base Model Analysis
    print("\nLoading BASE model...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        token=config.HUGGINGFACE_TOKEN,
        load_in_4bit=False
    )
    
    print("Tokenizing dataset...")
    test_tokenized = tokenize_dataset(test_dataset_raw, base_tokenizer)
    
    print("Computing BASE model loss...")
    base_loss = evaluate_manual_loss(base_model, test_tokenized)
    results["quantitative"]["base_loss"] = base_loss
    print(f"Base Loss: {base_loss:.4f}")

    audio_tokenizer = BiCodecTokenizer(config.SPARK_TTS_MODEL_DIR, "cuda")
    print("\nGenerating BASE audio samples...")
    for i, p in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {p}")
        generate_sample(base_model, base_tokenizer, audio_tokenizer, p, 
                        os.path.join(EVAL_DIR, "audio_base", f"sample_{i+1}.wav"))

    del base_model
    torch.cuda.empty_cache()

    # 4. Fine-tuned Model Analysis
    print("\nLoading FINE-TUNED model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNED_MODEL_PATH,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=False
    )

    print("Computing FINE-TUNED model loss...")
    ft_loss = evaluate_manual_loss(ft_model, test_tokenized)
    results["quantitative"]["finetuned_loss"] = ft_loss
    print(f"Fine-tuned Loss: {ft_loss:.4f}")

    print("\nGenerating FINE-TUNED audio samples...")
    for i, p in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {p}")
        generate_sample(ft_model, ft_tokenizer, audio_tokenizer, p, 
                        os.path.join(EVAL_DIR, "audio_finetuned", f"sample_{i+1}.wav"))

    # 5. Conclusion
    results["quantitative"]["improvement_pct"] = ((base_loss - ft_loss) / base_loss * 100) if base_loss > 0 else 0
    with open(os.path.join(EVAL_DIR, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print(f"Base Loss: {base_loss:.4f}")
    print(f"Fine-tuned Loss: {ft_loss:.4f}")
    print(f"Improvement: {results['quantitative']['improvement_pct']:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
