import json
import matplotlib.pyplot as plt
import os

# Load results
results_path = r'c:\Users\signb\Desktop\vishal\Finetuning_tts\Spark_TTS\outputs\final_results.json'
with open(results_path, 'r') as f:
    results = json.load(f)

# Extract values
val_loss = results['evaluation']['val_loss']
test_loss = results['evaluation']['test_loss']
# Train loss from training stats (last logged loss)
# We can estimate it or just use Val vs Test for clarity
# In this case, let's just do Val vs Test which is most important for generalization

categories = ['Validation Loss', 'Test Loss']
values = [val_loss, test_loss]
colors = ['#3498db', '#2ecc71']

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, values, color=colors, alpha=0.8)

# Add value labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(min(values) - 0.1, max(values) + 0.2)
plt.ylabel('Loss', fontsize=12)
plt.title('Final Model Performance Comparison', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

save_path = r'c:\Users\signb\Desktop\vishal\Finetuning_tts\Spark_TTS\outputs\test_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to {save_path}")
