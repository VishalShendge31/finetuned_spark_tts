"""
Step 4: Hyperparameter Tuning (Grid Search) for Spark-TTS
Memory-optimized for 12GB GPU (RTX 5070)
✅ COMPLETE FIXED VERSION - Custom Data Collator
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

from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
import config
from model_setup import setup_model
from utils import (plot_training_history, plot_hyperparameter_comparison,
                   save_hyperparameter_results, print_gpu_stats)
import json
from itertools import product
import gc
import time


def aggressive_memory_cleanup(model=None, tokenizer=None, stage=""):
    """
    Aggressive memory cleanup to prevent OOM errors
    Deletes models, runs garbage collection, and clears CUDA cache
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


def tokenize_dataset(dataset, tokenizer, max_length=2048):
    """
    Tokenize the dataset without creating labels
    Labels will be created by the custom collator
    
    ✅ FIXED: No label creation here - prevents nesting issues
    """
    def tokenize_function(examples):
        # Tokenize text only - no labels yet
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding in collator
            return_tensors=None,
        )
        return result
    
    print("\n🔤 Tokenizing dataset...")
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
    Create a custom data collator that properly handles:
    - Variable-length sequences
    - Padding to the longest sequence in batch
    - Label creation from input_ids
    - Masking padding tokens in labels
    
    ✅ FIXED: Custom collator prevents tensor shape mismatches
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    
    def collate_fn(features):
        """
        Collate function to batch examples together
        """
        # Extract input_ids and attention_mask as tensors
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in features]
        
        # Pad sequences to the same length
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
        
        # For causal language modeling, labels = input_ids
        labels = input_ids.clone()
        
        # Mask padding tokens in labels (set to -100 so they're ignored in loss)
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    return collate_fn


def load_existing_results():
    """Load existing hyperparameter search results if they exist"""
    results_file = os.path.join(config.HYPERPARAMETER_RESULTS_DIR, "results.json")
    
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"\n✓ Loaded {len(results)} existing results")
            return results
        except Exception as e:
            print(f"⚠️ Warning: Could not load results: {e}")
            return []
    return []


def is_run_completed(hyperparams, lora_r, results):
    """
    Check if a hyperparameter combination has already been tested
    Returns (is_completed, existing_result)
    """
    for result in results:
        # Skip OOM results
        if result.get('status') == 'OOM':
            continue
        
        # Check if all hyperparameters match
        if (result.get('learning_rate') == hyperparams.get('learning_rate') and
            result.get('gradient_accumulation_steps') == hyperparams.get('gradient_accumulation_steps') and
            result.get('num_train_epochs') == hyperparams.get('num_train_epochs') and
            result.get('lora_r') == lora_r):
            return True, result
    
    return False, None


def get_next_iteration_number(results):
    """Get the next iteration number for a new training run"""
    if not results:
        return 1
    return max([r.get('iteration', 0) for r in results]) + 1


def generate_grid_combinations():
    """
    Generate all hyperparameter combinations for grid search
    Uses config.HYPERPARAMETER_SEARCH_SPACE
    """
    param_names = list(config.HYPERPARAMETER_SEARCH_SPACE.keys())
    param_values = [config.HYPERPARAMETER_SEARCH_SPACE[name] for name in param_names]
    
    combinations = list(product(*param_values))
    
    print(f"\nGRID SEARCH:")
    for name, values in config.HYPERPARAMETER_SEARCH_SPACE.items():
        print(f"  - {name}: {values}")
    print(f"\n  Total combinations: {len(combinations)}")
    
    grid = []
    for combo in combinations:
        hyperparams = dict(zip(param_names, combo))
        grid.append(hyperparams)
    
    return grid


def train_with_hyperparameters(model, tokenizer, train_dataset, val_dataset, 
                               hyperparams, iteration, iteration_seed):
    """
    Train model with specific hyperparameters
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        hyperparams: Dictionary of hyperparameters
        iteration: Current iteration number
        iteration_seed: Random seed for this iteration
    
    Returns:
        eval_results, history, trainer_stats
    """
    
    # Extract hyperparameters
    batch_size = config.FIXED_BATCH_SIZE
    learning_rate = hyperparams.get('learning_rate')
    grad_accum = hyperparams.get('gradient_accumulation_steps')
    num_train_epochs = hyperparams.get('num_train_epochs', 1)
    warmup = config.FIXED_WARMUP_STEPS
    weight_decay = config.FIXED_WEIGHT_DECAY
    
    # Setup output directories
    output_dir = os.path.join(config.HYPERPARAMETER_RESULTS_DIR, f"run_{iteration}")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Tokenize datasets
    print("\n🔤 Tokenizing datasets...")
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer)
    
    # Calculate training steps
    steps_per_epoch = len(train_tokenized) // (batch_size * grad_accum)
    eval_steps = max(steps_per_epoch // 4, 1)
    
    # Print configuration
    print(f"\n⚙️  Configuration:")
    print(f"  Learning rate:        {learning_rate}")
    print(f"  Grad accumulation:    {grad_accum}")
    print(f"  Num epochs:           {num_train_epochs}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Steps per epoch:      ~{steps_per_epoch}")
    print(f"  Eval frequency:       {eval_steps} steps")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        seed=iteration_seed,
        fp16=False,  # Spark-TTS uses float32
        bf16=False,  # Spark-TTS uses float32
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to="tensorboard",
        logging_dir=log_dir,
        disable_tqdm=False,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Important for Windows
        dataloader_pin_memory=False,
        logging_first_step=True,
        logging_nan_inf_filter=True,
    )
    
    # Create custom data collator
    print("\n🔧 Creating custom data collator...")
    data_collator = create_custom_collator(tokenizer)
    print("✓ Custom collator created")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    print(f"\n🚀 Starting training...")
    
    try:
        # Train the model
        trainer_stats = trainer.train()
        
        # Print peak memory usage
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n📊 Peak VRAM: {peak:.2f} GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ OUT OF MEMORY ERROR")
            del trainer, data_collator, train_tokenized, val_tokenized
            raise
        raise
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    eval_results = trainer.evaluate()
    
    # Extract training history
    history = {
        'train_loss': [log['loss'] for log in trainer.state.log_history if 'loss' in log],
        'eval_loss': [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log],
        'learning_rate': [log['learning_rate'] for log in trainer.state.log_history if 'learning_rate' in log],
    }
    
    # Cleanup
    del trainer, data_collator, train_tokenized, val_tokenized
    
    return eval_results, history, trainer_stats


def main():
    """
    Main hyperparameter tuning pipeline
    - Loads dataset
    - Runs grid search over hyperparameters
    - Tracks best configuration
    - Saves results

    """
    print("="*70)
    print("Grid Search Hyperparameter Tuning (Spark-TTS)")
    print("MEMORY-OPTIMIZED FOR 12GB GPU")
    print("="*70)
    
    os.makedirs(config.HYPERPARAMETER_RESULTS_DIR, exist_ok=True)
    
    # Load existing results (for resuming)
    results = load_existing_results()
    
    # Load dataset
    print("\n📂 Loading dataset...")
    
    if not os.path.exists("processed_dataset"):
        print("❌ Error: Processed dataset not found!")
        print("   Run: python 2_data_preparation.py")
        return
    
    dataset_dict = load_from_disk("processed_dataset")
    train_dataset = dataset_dict["train"]
    val_dataset = dataset_dict["validation"]
    
    print(f"✓ Train: {len(train_dataset)} examples")
    print(f"✓ Validation: {len(val_dataset)} examples")
    
    print_gpu_stats()
    
    # Generate all hyperparameter combinations
    grid_combinations = generate_grid_combinations()
    total_combinations = len(grid_combinations)
    
    # Track best results
    best_eval_loss = float('inf')
    best_hyperparams = None
    best_history = None
    
    # Find current best from existing results
    if results:
        for result in results:
            if result.get('status') != 'OOM' and result.get('eval_loss', float('inf')) < best_eval_loss:
                best_eval_loss = result['eval_loss']
                best_hyperparams = result.copy()
        
        if best_hyperparams:
            print(f"\n📊 Current best eval_loss: {best_eval_loss:.4f}")
    
    # Statistics
    skipped = 0
    new_runs = 0
    failed_runs = 0
    
    # Grid search loop
    for idx, hyperparams in enumerate(grid_combinations):
        combo_number = idx + 1
        
        print("\n" + "="*70)
        print(f"Combination {combo_number}/{total_combinations}")
        print("="*70)
        
        lora_r = hyperparams.get('lora_r', config.LORA_R)
        
        # Check if already completed
        is_completed, existing_result = is_run_completed(hyperparams, lora_r, results)
        
        if is_completed:
            print(f"⏭️  SKIPPING - Already completed")
            print(f"   Previous eval_loss: {existing_result.get('eval_loss', 'N/A'):.4f}")
            skipped += 1
            continue
        
        new_runs += 1
        iteration_number = get_next_iteration_number(results)
        
        print(f"🆕 NEW RUN - Iteration {iteration_number}")
        
        # Set seed for reproducibility
        iteration_seed = config.RANDOM_SEED + iteration_number
        torch.manual_seed(iteration_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(iteration_seed)
        
        # Extract LoRA parameters
        current_hyperparams = hyperparams.copy()
        lora_r = current_hyperparams.pop('lora_r', config.LORA_R)
        lora_alpha = lora_r  # Match alpha to r
        
        print(f"\n🔧 Loading model (LoRA r={lora_r}, alpha={lora_alpha})...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model
        model, tokenizer = setup_model(lora_r=lora_r, lora_alpha=lora_alpha)
        
        try:
            # Train with current hyperparameters
            eval_results, history, trainer_stats = train_with_hyperparameters(
                model, tokenizer, train_dataset, val_dataset, 
                current_hyperparams, iteration_number, iteration_seed
            )
            
            # Store results
            result = {
                'iteration': iteration_number,
                'eval_loss': eval_results['eval_loss'],
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                **current_hyperparams,
                'train_runtime': trainer_stats.metrics['train_runtime'],
            }
            
            results.append(result)
            
            print(f"\n📊 Results:")
            print(f"   Eval loss:    {eval_results['eval_loss']:.4f}")
            print(f"   Train runtime: {trainer_stats.metrics['train_runtime']/60:.2f} min")
            
            # Check if this is the best result
            if eval_results['eval_loss'] < best_eval_loss:
                best_eval_loss = eval_results['eval_loss']
                best_hyperparams = result.copy()
                best_history = history
                print("   ⭐ NEW BEST RESULT!")
            
            # Save results immediately
            save_hyperparameter_results(results, 
                os.path.join(config.HYPERPARAMETER_RESULTS_DIR, "results.json"))
            
            # Plot training history for this run
            try:
                plot_training_history(history,
                    save_path=os.path.join(config.HYPERPARAMETER_RESULTS_DIR, f"history_run_{iteration_number}.png"))
            except Exception as e:
                print(f"⚠️ Warning: Could not save plot: {e}")
            
            plt.close('all')
            
        except Exception as e:
            error_msg = str(e)
            
            if "out of memory" in error_msg.lower():
                print(f"\n❌ OUT OF MEMORY ERROR - Skipping this configuration")
                
                result = {
                    'iteration': iteration_number,
                    'eval_loss': float('inf'),
                    'lora_r': lora_r,
                    'status': 'OOM',
                    **current_hyperparams
                }
                results.append(result)
                failed_runs += 1
                
                # Save results
                save_hyperparameter_results(results, 
                    os.path.join(config.HYPERPARAMETER_RESULTS_DIR, "results.json"))
                
                # Aggressive cleanup
                aggressive_memory_cleanup(model=model, tokenizer=tokenizer, 
                                        stage=f"ERROR Run {iteration_number}")
                model = None
                tokenizer = None
                continue
            else:
                print(f"\n❌ Unexpected error: {error_msg}")
                raise
        
        # Cleanup after successful run
        print(f"\n🧹 Cleanup after iteration {iteration_number}...")
        aggressive_memory_cleanup(model=model, tokenizer=tokenizer, 
                                stage=f"Post-run {iteration_number}")
        model = None
        tokenizer = None
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("🎉 HYPERPARAMETER TUNING COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Total combinations:   {total_combinations}")
    print(f"  Skipped (completed):  {skipped}")
    print(f"  New runs attempted:   {new_runs}")
    print(f"  Failed (OOM):         {failed_runs}")
    print(f"  Successful:           {new_runs - failed_runs}")
    
    if best_hyperparams:
        print(f"\n⭐ BEST CONFIGURATION (eval_loss={best_eval_loss:.4f}):")
        print("-" * 70)
        for k, v in best_hyperparams.items():
            if k not in ['iteration', 'train_runtime']:
                print(f"  {k:30s}: {v}")
        print("-" * 70)
        
        # Save best parameters
        best_path = os.path.join(config.HYPERPARAMETER_RESULTS_DIR, "best_hyperparameters.json")
        with open(best_path, 'w') as f:
            json.dump(best_hyperparams, f, indent=4)
        
        print(f"\n✅ Best parameters saved to: {best_path}")
        
        # Generate comparison plots
        successful_results = [r for r in results 
                            if r.get('status') != 'OOM' and r.get('eval_loss') != float('inf')]
        
        if len(successful_results) > 1:
            try:
                plot_hyperparameter_comparison(successful_results,
                    save_path=os.path.join(config.HYPERPARAMETER_RESULTS_DIR, "comparison.png"))
                print(f"✅ Comparison plot saved")
            except Exception as e:
                print(f"⚠️ Warning: Could not create comparison plot: {e}")
        
        if best_history:
            try:
                plot_training_history(best_history,
                    save_path=os.path.join(config.HYPERPARAMETER_RESULTS_DIR, "best_model_history.png"))
                print(f"✅ Best model history plot saved")
            except Exception as e:
                print(f"⚠️ Warning: Could not save best model plot: {e}")
    else:
        print("\n⚠️ No successful runs completed")
    
    print("\n" + "="*70)
    print("⏭️  NEXT STEP: python 5_train_full.py")
    print("   This will train the model with the best hyperparameters")
    print("="*70)


if __name__ == "__main__":
    """
    Entry point - sets seeds and runs main with error handling
    """
    # Set global seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    
    try:
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