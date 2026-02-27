"""
Utility functions for Spark-TTS fine-tuning
✅ GERMAN TEXT PREPROCESSING
"""
import torch
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import re
from typing import Dict, List
import numpy as np

# Import config AFTER all standard libraries

import config

def get_huggingface_token():
    """Get Hugging Face token from config"""
    if config.HUGGINGFACE_TOKEN:
        return config.HUGGINGFACE_TOKEN
    
    token = input("Enter your Hugging Face token (or press Enter to skip): ").strip()
    if token:
        config.HUGGINGFACE_TOKEN = token
        return token
    return None

# ============= GERMAN TEXT PREPROCESSING =============

def normalize_german_text(text):
    """
    Normalize German text for TTS
    ✅ UPDATED: Preserves non-verbal tokens like [sighs], [yawn], etc.
    
    - Lowercase text
    - Keep German umlauts (ä, ö, ü, ß)
    - Keep square brackets [] for non-verbal cues
    - Normalize numbers to German words
    - Remove other special characters

    """
    # Lowercase
    text = text.lower()
    
    # ✅ NEW: Keep square brackets and content inside them
    # Extract non-verbal tokens first
    non_verbal_pattern = r'\[[\w\-]+\]'
    non_verbal_tokens = re.findall(non_verbal_pattern, text)
    
    # Temporarily replace with placeholders
    placeholder_map = {}
    for i, token in enumerate(non_verbal_tokens):
        placeholder = f"NONVERBAL{i}"
        placeholder_map[placeholder] = token
        text = text.replace(token, placeholder, 1)
    
    # Keep German umlauts, basic punctuation, and our placeholders
    # ✅ FIXED: Now allows letters, umlauts, digits, punctuation, and uppercase letters (for placeholders)
    text = re.sub(r'[^a-zäöüßA-Z\s\d.,!?\-<>]', '', text)
    
    # Restore non-verbal tokens
    for placeholder, original in placeholder_map.items():
        text = text.replace(placeholder, original)
    
    # Normalize numbers to German words
    try:
        from num2words import num2words
        
        def replace_number(match):
            try:
                num = int(match.group())
                return num2words(num, lang='de')
            except:
                return match.group()
        
        # Only replace numbers that are NOT inside brackets
        def replace_if_not_in_brackets(text):
            result = []
            i = 0
            while i < len(text):
                if text[i] == '[':
                    # Find closing bracket
                    end = text.find(']', i)
                    if end != -1:
                        result.append(text[i:end+1])
                        i = end + 1
                    else:
                        result.append(text[i])
                        i += 1
                elif text[i].isdigit():
                    # Extract full number
                    j = i
                    while j < len(text) and text[j].isdigit():
                        j += 1
                    num = int(text[i:j])
                    result.append(num2words(num, lang='de'))
                    i = j
                else:
                    result.append(text[i])
                    i += 1
            return ''.join(result)
        
        text = replace_if_not_in_brackets(text)
        
    except ImportError:
        print("⚠️ Warning: num2words not installed. Numbers will not be normalized.")
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def validate_german_text(text):
    """
    Validate that German text is properly formatted
    ✅ UPDATED: Allows square brackets for non-verbal tokens
    """
    # Allow: letters, umlauts, spaces, punctuation, square brackets, underscores
    invalid_pattern = r'[^a-zäöüß\s.,!?\-<>\[\]_]'
    has_invalid = bool(re.search(invalid_pattern, text))
    return not has_invalid

# ============= VISUALIZATION =============

def plot_training_history(history: Dict, save_path: str = None):
    """Plot training and validation loss"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if 'train_loss' in history and history['train_loss']:
        ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    if 'eval_loss' in history and history['eval_loss']:
        ax.plot(history['eval_loss'], label='Validation Loss', linewidth=2, marker='o')
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()

def plot_hyperparameter_comparison(results, save_path: str = None):
    """Plot comparison of hyperparameter configurations"""
    import pandas as pd
    
    df = pd.DataFrame(results)
    df = df.sort_values('eval_loss')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Spark-TTS Training Analysis', fontsize=16, fontweight='bold')
    
    top_n = min(10, len(df))
    top_df = df.head(top_n)
    
    x_labels = [f"Config {i+1}" for i in range(top_n)]
    
    axes[0].bar(x_labels, top_df['eval_loss'], color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Top Hyperparameter Configurations', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    hyperparam_cols = ['learning_rate', 'gradient_accumulation_steps', 'lora_r']
    hyperparam_cols = [col for col in hyperparam_cols if col in top_df.columns]
    
    if hyperparam_cols:
        heatmap_data = top_df[hyperparam_cols].values.T
        sns.heatmap(heatmap_data, annot=True, fmt='.2e', cmap='YlOrRd',
                   xticklabels=x_labels, yticklabels=hyperparam_cols,
                   ax=axes[1], cbar_kws={'label': 'Value'})
        axes[1].set_title('Hyperparameter Values', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()
    
    return df

def save_hyperparameter_results(results, filepath: str):
    """Save hyperparameter search results to JSON"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")

def load_hyperparameter_results(filepath: str):
    """Load hyperparameter search results from JSON"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        results = json.load(f)
    print(f"Results loaded from {filepath}")
    return results

def print_gpu_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        memory_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"\nGPU: {gpu_stats.name}")
        print(f"Max memory: {max_memory} GB")
        print(f"Reserved memory: {memory_reserved} GB")
        print(f"Memory usage: {round(memory_reserved / max_memory * 100, 2)}%")
    else:
        print("\nNo GPU available")