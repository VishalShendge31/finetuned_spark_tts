"""
Step 5: Full training with best hyperparameters
Automatic mode for Spark-TTS
✅ FIXED VERSION - Consistent with hyperparameter tuning pipeline
"""
import setup_environment
import os
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["UNSLOTH_DISABLE_XFORMERS"] = "1"
os.environ["UNSLOTH_ATTENTION"] = "torch"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('Spark-TTS')

import torch
import json
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
import config
from model_setup import setup_model
from utils import plot_training_history, print_gpu_stats
import gc
import time

AUTO_MODE = True
AUTO_SAVE_MERGED = False


def aggressive_memory_cleanup(model=None, tokenizer=None, stage=""):
    """
    Aggressive memory cleanup to prevent OOM errors
    ✅ Same as hyperparameter_tuning.py
    """
    print(f"\n{'='*60}")
    print(f"MEMORY CLEANUP - {stage}")
    print(f"{'='*60}")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"Before cleanup: {allocated:.2f} GB allocated")
    
    # Delete model
    if model is not None:
        try:
            model.to('cpu')
            del model
        except:
            pass
    
    # Delete tokenizer
    if tokenizer is not None:
        try:
            del tokenizer
        except:
            pass
    
    # Close all matplotlib figures
    plt.close('all')
    
    # Multiple garbage collection passes
    for _ in range(5):
        gc.collect()
    
    # CUDA cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    time.sleep(2)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"After cleanup: {allocated:.2f} GB allocated")
    
    print("Cleanup complete\n")


def load_best_hyperparameters():
    """Load best hyperparameters from tuning"""
    best_hyperparams_path = os.path.join(config.HYPERPARAMETER_RESULTS_DIR, 
                                         "best_hyperparameters.json")
    
    if not os.path.exists(best_hyperparams_path):
        print("⚠️  Warning: Best hyperparameters not found. Using default config.")
        
        if AUTO_MODE:
            print("    ✅ AUTO_MODE enabled: Continuing with default config")
            return None
        
        response = input("\nContinue with default config? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting.")
            exit(0)
        return None
    
    with open(best_hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)
    
    print("\n📋 Loaded best hyperparameters:")
    for key, value in best_hyperparams.items():
        if key not in ['iteration', 'train_runtime', 'status']:
            print(f"  {key}: {value}")
    
    return best_hyperparams


def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """
    Tokenize the dataset without creating labels
    Labels will be created by the custom collator
    ✅ Same as hyperparameter_tuning.py
    """
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        return result
    
    print("🔤 Tokenizing...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        batch_size=100,
    )
    print("✓ Tokenization complete")
    
    return tokenized


def create_custom_collator(tokenizer):
    """
    Create custom data collator
    ✅ Same as hyperparameter_tuning.py - prevents tensor shape issues
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    
    def collate_fn(features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        
        input_ids = pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0
        )
        
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    return collate_fn


def create_detailed_plots(history, output_dir):
    """Create detailed training visualization plots"""
    try:
        print("\n📊 Creating visualizations...")
        
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, "training_history.png")
        )
        print("  ✓ training_history.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Spark-TTS Full Training Analysis', fontsize=16, fontweight='bold')
        
        # Training loss
        if history['train_loss']:
            axes[0, 0].plot(history['train_loss'], alpha=0.6, linewidth=1)
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Validation loss
        if history['eval_loss']:
            axes[0, 1].plot(history['eval_loss'], marker='o', linewidth=2)
            best_loss = min(history['eval_loss'])
            best_idx = history['eval_loss'].index(best_loss)
            axes[0, 1].axhline(y=best_loss, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].plot(best_idx, best_loss, 'r*', markersize=15)
            axes[0, 1].set_xlabel('Evaluation Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].text(best_idx, best_loss, f' {best_loss:.4f}', 
                          fontsize=9, va='bottom')
        
        # Learning rate
        if history['learning_rate']:
            axes[1, 0].plot(history['learning_rate'], color='green', linewidth=2)
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined
        if history['train_loss'] and history['eval_loss']:
            eval_interval = len(history['train_loss']) // len(history['eval_loss']) if history['eval_loss'] else 1
            eval_steps = [i * eval_interval for i in range(len(history['eval_loss']))]
            
            axes[1, 1].plot(history['train_loss'], alpha=0.6, label='Training')
            axes[1, 1].plot(eval_steps, history['eval_loss'], marker='o', label='Validation', linewidth=2)
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training vs Validation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(output_dir, "training_detailed.png"), dpi=150)
        plt.close()
        print(f"  ✓ training_detailed.png")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create plots: {e}")


def main():
    print("=" * 70)
    print("Step 5: Full Training (Spark-TTS)")
    if AUTO_MODE:
        print("🤖 AUTOMATIC MODE")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    
    if not os.path.exists("processed_dataset"):
        print("❌ Error: Processed dataset not found!")
        print("   Please run 2_data_preparation.py first")
        return
    
    dataset_dict = load_from_disk("processed_dataset")
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    test_dataset = dataset_dict["test"]
    
    print(f"✓ Train: {len(train_dataset)}")
    print(f"✓ Validation: {len(val_dataset)}")
    print(f"✓ Test: {len(test_dataset)}")
    
    # Load best hyperparameters
    print("\n🔍 Loading Hyperparameters...")
    
    best_hyperparams = load_best_hyperparameters()
    
    if best_hyperparams:
        lora_r = best_hyperparams.get('lora_r', config.LORA_R)
        lora_alpha = best_hyperparams.get('lora_alpha', config.LORA_ALPHA)
        learning_rate = best_hyperparams.get('learning_rate', config.TRAINING_CONFIG['learning_rate'])
        batch_size = best_hyperparams.get('per_device_train_batch_size', config.FIXED_BATCH_SIZE)
        grad_accum = best_hyperparams.get('gradient_accumulation_steps', 
                                          config.TRAINING_CONFIG['gradient_accumulation_steps'])
        num_train_epochs = best_hyperparams.get('num_train_epochs', 3)
        warmup = best_hyperparams.get('warmup_steps', config.FIXED_WARMUP_STEPS)
        weight_decay = best_hyperparams.get('weight_decay', config.FIXED_WEIGHT_DECAY)
        print("✓ Using tuned hyperparameters")
    else:
        lora_r = config.LORA_R
        lora_alpha = config.LORA_ALPHA
        learning_rate = config.TRAINING_CONFIG['learning_rate']
        batch_size = config.TRAINING_CONFIG['per_device_train_batch_size']
        grad_accum = config.TRAINING_CONFIG['gradient_accumulation_steps']
        num_train_epochs = 3
        warmup = config.TRAINING_CONFIG['warmup_steps']
        weight_decay = config.TRAINING_CONFIG['weight_decay']
        print("✓ Using default hyperparameters")
    
    # Setup model
    print("\n🔧 Initializing Model...")
    print(f"   LoRA rank: {lora_r}")
    print(f"   LoRA alpha: {lora_alpha}")
    
    aggressive_memory_cleanup(stage="Pre-model load")
    
    model, tokenizer = setup_model(lora_r=lora_r, lora_alpha=lora_alpha)
    
    print_gpu_stats()
    
    # ✅ CRITICAL FIX: Tokenize datasets BEFORE passing to Trainer
    print("\n🔤 Preparing datasets...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, max_length=config.MAX_SEQ_LENGTH)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, max_length=config.MAX_SEQ_LENGTH)
    test_tokenized = tokenize_dataset(test_dataset, tokenizer, max_length=config.MAX_SEQ_LENGTH)
    
    # Training configuration
    print("\n⚙️  Training Configuration...")
    
    steps_per_epoch = len(train_tokenized) // (batch_size * grad_accum)
    eval_steps = max(steps_per_epoch // 5, 1)
    save_steps = eval_steps
    
    print(f"\n📊 Training setup:")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Dataset size: {len(train_tokenized)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Steps per epoch: ~{steps_per_epoch}")
    print(f"  Eval frequency: {eval_steps} steps")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup}")
    print(f"  Weight decay: {weight_decay}")
    
    # Create output directory
    log_dir = os.path.join(config.OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=config.RANDOM_SEED,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="none",
        logging_dir=log_dir,
        disable_tqdm=False,
        max_grad_norm=1.0,
        remove_unused_columns=False,  # ✅ Keep this False with custom collator
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        logging_first_step=True,
        logging_nan_inf_filter=True,
    )
    
    # ✅ CRITICAL FIX: Use custom collator (NOT DataCollatorForSeq2Seq)
    print("\n🔧 Creating custom data collator...")
    data_collator = create_custom_collator(tokenizer)
    print("✓ Custom collator created")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,  # ✅ Use tokenized dataset
        eval_dataset=val_tokenized,     # ✅ Use tokenized dataset
        data_collator=data_collator,     # ✅ Use custom collator
    )
    
    # Training
    print("\n🚀 Starting Training...")
    
    if AUTO_MODE:
        print("🤖 AUTO_MODE: Starting immediately...")
    else:
        input("\n▶️  Press Enter to start...")
    
    # Record start memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / 1024**3
    else:
        start_memory = 0
    
    # Train
    try:
        print("\n🏋️  Training...\n")
        trainer_stats = trainer.train()
        
        print("\n✅ Training Complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        interrupted_dir = os.path.join(config.OUTPUT_DIR, "interrupted_model")
        trainer.save_model(interrupted_dir)
        tokenizer.save_pretrained(interrupted_dir)
        print(f"✓ Model saved to: {interrupted_dir}")
        
        # Save training state for resumption
        trainer.save_state()
        print(f"✓ Training state saved - you can resume training later")
        return
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ OUT OF MEMORY ERROR")
            print("\n💡 Solutions:")
            print("   1. Reduce per_device_train_batch_size in best hyperparameters")
            print("   2. Increase gradient_accumulation_steps")
            print("   3. Reduce MAX_SEQ_LENGTH in config.py")
            print("   4. Try smaller LoRA rank")
            
            # Emergency save
            try:
                emergency_dir = os.path.join(config.OUTPUT_DIR, "emergency_checkpoint")
                trainer.save_model(emergency_dir)
                print(f"✓ Emergency checkpoint saved to: {emergency_dir}")
            except:
                pass
            
            torch.cuda.empty_cache()
            return
        else:
            print(f"\n❌ Runtime error: {e}")
            raise e
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Statistics
    print("\n📈 Training Statistics:")
    print(f"  Runtime: {trainer_stats.metrics['train_runtime']/60:.2f} min")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak GPU memory: {peak_memory:.3f} GB")
    
    # Evaluate - ✅ Use tokenized test dataset
    print("\n📊 Evaluating...")
    
    try:
        val_results = trainer.evaluate(eval_dataset=val_tokenized)
        print(f"✓ Validation loss: {val_results['eval_loss']:.4f}")
        
        test_results = trainer.evaluate(eval_dataset=test_tokenized)
        print(f"✓ Test loss: {test_results['eval_loss']:.4f}")
    except Exception as e:
        print(f"⚠️  Evaluation error: {e}")
        val_results = {'eval_loss': float('nan')}
        test_results = {'eval_loss': float('nan')}
    
    # Save results
    print("\n💾 Saving Results...")
    
    results = {
        'hyperparameters': {
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': grad_accum,
            'num_train_epochs': num_train_epochs,
            'warmup_steps': warmup,
            'weight_decay': weight_decay,
        },
        'evaluation': {
            'val_loss': val_results['eval_loss'],
            'test_loss': test_results['eval_loss'],
        },
        'training': {
            'runtime_minutes': trainer_stats.metrics['train_runtime'] / 60,
            'peak_memory_gb': peak_memory if torch.cuda.is_available() else None,
        },
        'metadata': {
            'dataset_train_size': len(train_dataset),
            'dataset_val_size': len(val_dataset),
            'dataset_test_size': len(test_dataset),
            'model_name': config.MODEL_NAME,
            'random_seed': config.RANDOM_SEED,
        }
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Results saved: {results_path}")
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    
    try:
        history = {
            'train_loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log],
            'eval_loss': [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log],
            'learning_rate': [log['learning_rate'] for log in trainer.state.log_history if 'learning_rate' in log],
        }
        
        create_detailed_plots(history, config.OUTPUT_DIR)
    except Exception as e:
        print(f"⚠️  Warning: Could not create plots: {e}")
    
    # Save model
    print("\n💾 Saving Model...")
    
    try:
        model.save_pretrained(config.BEST_MODEL_DIR)
        tokenizer.save_pretrained(config.BEST_MODEL_DIR)
        print(f"✓ Model saved: {config.BEST_MODEL_DIR}")
        
        # Save training state for potential resumption
        trainer.save_state()
        print(f"✓ Training state saved")
        
        # Save training arguments
        training_args_path = os.path.join(config.BEST_MODEL_DIR, "training_args.json")
        with open(training_args_path, 'w') as f:
            json.dump(training_args.to_dict(), f, indent=4)
        print(f"✓ Training arguments saved")
        
    except Exception as e:
        print(f"⚠️  Warning: Error saving model: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"  ✓ Validation loss: {val_results['eval_loss']:.4f}")
    print(f"  ✓ Test loss: {test_results['eval_loss']:.4f}")
    print(f"  ✓ Training time: {trainer_stats.metrics['train_runtime']/60:.2f} min")
    if torch.cuda.is_available():
        print(f"  ✓ Peak memory: {peak_memory:.2f} GB")
    
    print(f"\n📁 Outputs:")
    print(f"  ✓ Model: {config.BEST_MODEL_DIR}")
    print(f"  ✓ Results: {results_path}")
    print(f"  ✓ Plots: {config.OUTPUT_DIR}/training_*.png")
    print(f"  ✓ Logs: {log_dir}")
    
    print("\n🎯 Model Performance:")
    improvement = ((val_results['eval_loss'] - test_results['eval_loss']) / val_results['eval_loss'] * 100)
    print(f"  Val-Test diff: {improvement:+.2f}%")
    
    if abs(improvement) < 5:
        print("  ✅ Good generalization (val ≈ test)")
    elif improvement > 10:
        print("  ⚠️  Possible overfitting (val << test)")
    else:
        print("  ✅ Acceptable performance")
    
    print("\n" + "="*70)
    print("⏭️  Next step: python 6_inference.py")
    print("   Test your trained model with custom inputs!")
    print("="*70)


if __name__ == "__main__":
    try:
        # Set seeds for reproducibility
        torch.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_SEED)
        
        main()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Final cleanup...")
        aggressive_memory_cleanup(stage="End")
        print("✅ Done!")