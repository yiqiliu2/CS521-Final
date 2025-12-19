import os
import re
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
dir_a = os.path.join(base_dir, 'outputs_icbm_4096_a')
dir_b = os.path.join(base_dir, 'outputs_icbm_4096_b')

data = []

def parse_icbm_ms(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Look for "prepare time: <number> sec"
            match = re.search(r'prepare time:\s*([\d\.]+)\s*sec', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

# Walk through dir_a
for root, dirs, files in os.walk(dir_a):
    for file in files:
        if file.endswith(".log"):
            # Get relative path from dir_a
            rel_path = os.path.relpath(os.path.join(root, file), dir_a)
            path_a = os.path.join(dir_a, rel_path)
            path_b = os.path.join(dir_b, rel_path)
            
            if os.path.exists(path_b):
                val_a = parse_icbm_ms(path_a)
                val_b = parse_icbm_ms(path_b)
                
                if val_a is not None and val_b is not None:
                    # Extract core count and config name
                    # rel_path is like "2208cores/gemma2-b32-6000GBps.log"
                    parts = rel_path.split('/')
                    if len(parts) == 2:
                        core_str = parts[0]
                        config_name = parts[1].replace('.log', '')
                        
                        # Try to parse core count
                        core_match = re.match(r'(\d+)cores', core_str)
                        if core_match:
                            cores = int(core_match.group(1))
                            data.append({
                                'cores': cores,
                                'config': config_name,
                                'val_a': val_a,
                                'val_b': val_b
                            })

print(f"Found {len(data)} matching comparisons.")

# Analyze B vs A
better_count = 0
total_count = len(data)
speedups = []
b2_data = []

print(f"{'Config':<50} {'A (sec)':<10} {'B (sec)':<10} {'Speedup (A/B)':<10}")
print("-" * 85)

for d in data:
    speedup = d['val_a'] / d['val_b'] if d['val_b'] > 0 else 0
    speedups.append(speedup)
    if d['val_b'] < d['val_a']:
        better_count += 1
    
    # Check for "b2" in config name (assuming it means batch 2, usually formatted like -b2-)
    # or just "b2" substring
    if "b2" in d['config'] and "b32" not in d['config'] and "b12" not in d['config'] and "b22" not in d['config']: 
        # Crude check to avoid matching b32 as b2. Better to check segments.
        # But looking at file names like gemma2-b32, b2 would be gemma2-b2.
        b2_data.append(d)

    print(f"{d['config']:<50} {d['val_a']:<10.4f} {d['val_b']:<10.4f} {speedup:<10.4f}")

print("-" * 85)
print(f"B is better (lower time) in {better_count}/{total_count} cases ({better_count/total_count*100:.1f}%).")
if speedups:
    print(f"Average Speedup (A/B): {sum(speedups)/len(speedups):.4f}")
    print(f"Max Speedup: {max(speedups):.4f}")
    print(f"Min Speedup: {min(speedups):.4f}")

if b2_data:
    print("\n--- Analysis for 'b2' configurations ---")
    for d in b2_data:
        speedup = d['val_a'] / d['val_b'] if d['val_b'] > 0 else 0
        print(f"{d['config']:<50} A: {d['val_a']:.4f} B: {d['val_b']:.4f} Speedup: {speedup:.4f}")
else:
    print("\nNo specific 'b2' configurations found (checked for 'b2' excluding 'b32' etc).")

# Group by model to make plots readable
# Config name usually starts with model name, e.g., "gemma2-b32-..."
def get_model_name(config_name):
    return config_name.split('-')[0]

data.sort(key=lambda x: (get_model_name(x['config']), x['cores']))

models = sorted(list(set(get_model_name(d['config']) for d in data)))

# Create a plot for each model
for model in models:
    model_data = [d for d in data if get_model_name(d['config']) == model]
    
    if not model_data:
        continue
        
    # Prepare data for plotting
    labels = [f"{d['cores']}c\n{d['config']}" for d in model_data]
    # Simplify labels: just show cores and maybe bandwidth if it varies
    # Or just use an index and a legend
    
    # Let's try a scatter plot: A vs B
    # Or a bar chart side-by-side
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    indices = np.arange(len(model_data))
    width = 0.35
    
    vals_a = [d['val_a'] for d in model_data]
    vals_b = [d['val_b'] for d in model_data]
    
    rects1 = ax.bar(indices - width/2, vals_a, width, label='A')
    rects2 = ax.bar(indices + width/2, vals_b, width, label='B')
    
    ax.set_ylabel('prepare time (sec)')
    ax.set_title(f'Comparison of prepare time for {model} (A vs B)')
    ax.set_xticks(indices)
    
    # X labels might be too crowded. Let's try to make them concise.
    # config name: "gemma2-b32-6000GBps" -> "6000GBps" if model is gemma2
    x_labels = []
    for d in model_data:
        lbl = d['config'].replace(model, '').strip('-')
        lbl = f"{d['cores']}c\n{lbl}"
        x_labels.append(lbl)
        
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'compare_prepare_{model}.png')
    print(f"Saved compare_prepare_{model}.png")

# Also a global scatter plot
fig, ax = plt.subplots(figsize=(10, 10))
all_a = [d['val_a'] for d in data]
all_b = [d['val_b'] for d in data]

ax.scatter(all_a, all_b)
ax.plot([0, max(all_a + all_b)], [0, max(all_a + all_b)], 'r--') # Diagonal line
ax.set_xlabel('prepare time (A)')
ax.set_ylabel('prepare time (B)')
ax.set_title('Global Comparison: prepare time A vs B')
plt.savefig('compare_prepare_global.png')
print("Saved compare_prepare_global.png")
