"""
Step 2: Load and prepare the dataset for Spark-TTS
- Load dataset from Hugging Face
- Tokenize audio using BiCodec
- German text preprocessing
- Stratified split by emotion (80/10/10)

"""
import sys
import os

# Add Spark-TTS to path

sys.path.append('Spark-TTS')

import torch
import numpy as np
from datasets import load_dataset, DatasetDict
import torchaudio.transforms as T

# Import config and utils

import config
import utils

# Import Spark-TTS components

from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize


def extract_wav2vec2_features(audio_tokenizer, wavs: torch.Tensor) -> torch.Tensor:
    """Extract wav2vec2 features from audio"""
    if wavs.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")
    
    wav_np = wavs.squeeze(0).cpu().numpy()
    
    # Process with wav2vec2
    processed = audio_tokenizer.processor(
        wav_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = processed.input_values.to(audio_tokenizer.feature_extractor.device)
    
    # Extract features
    model_output = audio_tokenizer.feature_extractor(input_values)
    
    if model_output.hidden_states is None:
        raise ValueError("Wav2Vec2Model did not return hidden states.")
    
    # Mix layers 11, 14, 16 as per Spark-TTS
    feats_mix = (
        model_output.hidden_states[11] + 
        model_output.hidden_states[14] + 
        model_output.hidden_states[16]
    ) / 3
    
    return feats_mix


def formatting_audio_func(example, audio_tokenizer):
    """Convert audio to Spark-TTS format with German text preprocessing"""
    try:
        # German text preprocessing
        text = example.get('text', '')
        text = utils.normalize_german_text(text)
        
        if not utils.validate_german_text(text):
            print(f"⚠️ Warning: Text validation failed: {text[:50]}...")
        
        # Add emotion prefix if available
        if "source" in example and example["source"]:
            text = f"{example['source']}: {text}"
        
        # Process audio
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        
        target_sr = audio_tokenizer.config['sample_rate']
        
        # Resample if needed
        if sampling_rate != target_sr:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
            audio_tensor_temp = torch.from_numpy(audio_array).float()
            audio_array = resampler(audio_tensor_temp).numpy()
        
        # Volume normalize
        if audio_tokenizer.config["volume_normalize"]:
            audio_array = audio_volume_normalize(audio_array)
        
        # Get reference clip
        ref_wav_np = audio_tokenizer.get_ref_clip(audio_array)
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(audio_tokenizer.device)
        ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(audio_tokenizer.device)
        
        # Extract features
        feat = extract_wav2vec2_features(audio_tokenizer, audio_tensor)
        
        batch = {
            "wav": audio_tensor,
            "ref_wav": ref_wav_tensor,
            "feat": feat.to(audio_tokenizer.device),
        }
        
        # Tokenize with BiCodec
        semantic_token_ids, global_token_ids = audio_tokenizer.model.tokenize(batch)
        
        # Format tokens as strings
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()]
        )
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()]
        )
        
        # Create input format for Spark-TTS
        inputs = [
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
            "<|end_semantic_token|>",
            "<|im_end|>"
        ]
        inputs = "".join(inputs)
        
        return {"text": inputs}
    
    except Exception as e:
        print(f"❌ Error processing example: {e}")
        return {"text": None}


def main():
    """Main data preparation pipeline"""
    print("=" * 70)
    print("Step 2: Data Preparation (Spark-TTS + German)")
    print("✅ GERMAN TEXT PREPROCESSING ENABLED")
    print("=" * 70)
    
    # Get Hugging Face token
    print("\n🔑 Hugging Face Authentication")
    hf_token = utils.get_huggingface_token()
    
    if not hf_token:
        print("⚠️ Warning: No Hugging Face token provided")
        print("   Some datasets may not be accessible")
    
    # Load dataset
    print(f"\n📂 Loading dataset: {config.DATASET_NAME}")
    print("   This may take a few minutes...")
    
    try:
        dataset = load_dataset(config.DATASET_NAME, split="train", token=hf_token)
        print(f"✓ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check dataset structure
    print(f"\n📊 Dataset info:")
    print(f"   Columns: {dataset.column_names}")
    print(f"   Features: {dataset.features}")
    
    # Load BiCodec tokenizer
    print(f"\n🔊 Loading BiCodec tokenizer...")
    print(f"   Model path: {config.SPARK_TTS_MODEL_DIR}")
    
    if not os.path.exists(config.SPARK_TTS_MODEL_DIR):
        print(f"❌ Error: Spark-TTS model not found at {config.SPARK_TTS_MODEL_DIR}")
        print("   Please run: python 1_download_model.py")
        return
    
    try:
        audio_tokenizer = BiCodecTokenizer(config.SPARK_TTS_MODEL_DIR, "cuda")
        print("✓ BiCodec tokenizer loaded")
    except Exception as e:
        print(f"❌ Error loading BiCodec: {e}")
        return
    
    utils.print_gpu_stats()
    
    # Process dataset
    print("\n🔄 Processing dataset...")
    print("   This will take several minutes...")
    print("   Operations:")
    print("   • Resampling audio to 16kHz")
    print("   • Extracting Wav2Vec2 features")
    print("   • Tokenizing with BiCodec")
    print("   • Normalizing German text")
    
    try:
        dataset = dataset.map(
            lambda x: formatting_audio_func(x, audio_tokenizer),
            remove_columns=["audio"],
            desc="Processing audio"
        )
        print("✓ Dataset processing complete")
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return
    
    # Filter out failed samples
    original_len = len(dataset)
    dataset = dataset.filter(
        lambda x: x["text"] is not None and len(x["text"]) > 0,
        desc="Filtering valid samples"
    )
    filtered_len = len(dataset)
    
    if filtered_len < original_len:
        print(f"⚠️ Filtered out {original_len - filtered_len} failed samples")
    print(f"✓ Final dataset: {filtered_len} examples")
    
    # Move models to CPU to free GPU memory
    print("\n🧹 Freeing GPU memory...")
    audio_tokenizer.model.cpu()
    audio_tokenizer.feature_extractor.cpu()
    del audio_tokenizer
    torch.cuda.empty_cache()
    print("✓ GPU memory freed")
    
    # Dataset splitting
    print(f"\n" + "=" * 70)
    print("📊 Dataset Splitting (Stratified by Emotion)")
    print("=" * 70)
    
    # Reload original dataset for emotion labels
    original_dataset = load_dataset(config.DATASET_NAME, split="train", token=hf_token)
    
    if "source" in original_dataset.column_names:
        print("\n📊 Emotion distribution:")
        
        # Count emotions
        emotion_counts = {}
        for example in original_dataset:
            emotion = example.get("source", "unknown")
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_samples = sum(emotion_counts.values())
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / total_samples) * 100
            print(f"  {emotion:20s}: {count:4d} samples ({percentage:5.1f}%)")
        
        # Add stratification column
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(emotion_counts.keys()))}
        
        emotion_indices = []
        for i in range(len(original_dataset)):
            if i < len(dataset):
                emotion = original_dataset[i].get("source", "unknown")
                emotion_indices.append(emotion_to_idx.get(emotion, 0))
        
        dataset = dataset.add_column("stratify_column", emotion_indices)
        
        # Stratified split
        print("\nPerforming stratified split...")
        train_test = dataset.train_test_split(
            test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
            seed=config.RANDOM_SEED,
            stratify_by_column="stratify_column"
        )
        
        val_test_size = config.VAL_SPLIT + config.TEST_SPLIT
        test_size = config.TEST_SPLIT / val_test_size
        
        val_test = train_test["test"].train_test_split(
            test_size=test_size,
            seed=config.RANDOM_SEED,
            stratify_by_column="stratify_column"
        )
        
        train_dataset = train_test["train"].remove_columns(["stratify_column"])
        val_dataset = val_test["train"].remove_columns(["stratify_column"])
        test_dataset = val_test["test"].remove_columns(["stratify_column"])
        
        print("✓ Stratified split complete")
    else:
        print("\n⚠️ 'source' column not found - using random split")
        
        train_test = dataset.train_test_split(
            test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
            seed=config.RANDOM_SEED
        )
        
        val_test_size = config.VAL_SPLIT + config.TEST_SPLIT
        test_size = config.TEST_SPLIT / val_test_size
        
        val_test = train_test["test"].train_test_split(
            test_size=test_size,
            seed=config.RANDOM_SEED
        )
        
        train_dataset = train_test["train"]
        val_dataset = val_test["train"]
        test_dataset = val_test["test"]
    
    # Create final dataset dictionary
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    print(f"\n✅ Final dataset splits:")
    print(f"  Train:      {len(dataset_dict['train']):4d} examples ({config.TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation: {len(dataset_dict['validation']):4d} examples ({config.VAL_SPLIT*100:.0f}%)")
    print(f"  Test:       {len(dataset_dict['test']):4d} examples ({config.TEST_SPLIT*100:.0f}%)")
    
    # Save processed dataset
    save_path = "processed_dataset"
    print(f"\n💾 Saving processed dataset...")
    print(f"   Location: {save_path}")
    
    try:
        dataset_dict.save_to_disk(save_path)
        print("✓ Dataset saved successfully")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 DATA PREPARATION COMPLETE!")
    print("=" * 70)
    print("\n✅ Completed tasks:")
    print("  • Loaded dataset from Hugging Face")
    print("  • Preprocessed German text")
    print("  • Tokenized audio with BiCodec")
    print("  • Extracted Wav2Vec2 features")
    print("  • Split dataset (stratified by emotion)")
    print("  • Saved processed dataset")
    
    print(f"\n📁 Processed dataset saved to: {save_path}")
    print(f"   Train samples:      {len(dataset_dict['train'])}")
    print(f"   Validation samples: {len(dataset_dict['validation'])}")
    print(f"   Test samples:       {len(dataset_dict['test'])}")
    
    print("\n⏭️  NEXT STEP: python 3_model_setup.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()