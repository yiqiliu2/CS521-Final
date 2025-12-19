import os
import re
import matplotlib.pyplot as plt
import collections
import numpy as np

# Set larger font sizes
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

base_dir = './benchmark_scripts/outputs_a'

data = []

def parse_file(filepath):
    prepare_time = None
    icbm_time = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "prepare time:" in line:
                    match = re.search(r'prepare time:\s*([\d\.]+)', line)
                    if match:
                        prepare_time = float(match.group(1))
                if "icbm_ordered schedule time:" in line:
                    match = re.search(r'icbm_ordered schedule time:\s*([\d\.]+)', line)
                    if match:
                        icbm_time = float(match.group(1))
    except Exception:
        return None, None
    return prepare_time, icbm_time

for model in os.listdir(base_dir):
    model_dir = os.path.join(base_dir, model)
    if not os.path.isdir(model_dir):
        continue
    
    for filename in os.listdir(model_dir):
        if not filename.startswith("test-bw-"):
            continue
        
        parts = filename.split('-')
        if len(parts) < 6:
            continue
            
        try:
            batch = int(parts[2])
            seq = int(parts[3])
            numcore = int(parts[5])
        except (IndexError, ValueError):
            continue

        filepath = os.path.join(model_dir, filename)
        prepare_time, icbm_time = parse_file(filepath)
        
        if prepare_time is not None:
            # Filter out outliers (specifically high ICBM time for llama2-13 at batch 2)
            if icbm_time is not None and icbm_time > 40:
                continue
                
            data.append({
                'model': model,
                'batch': batch,
                'seq': seq,
                'numcore': numcore,
                'prepare_time': prepare_time,
                'icbm_time': icbm_time
            })

# Group by model
models = sorted(list(set(d['model'] for d in data)))
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

# --- Batch Sensitivity ---
# Use seq=2048 for llama2-13 and opt, seq=4096 for others
fig, ax = plt.subplots(figsize=(12, 8))

for model, color in zip(models, colors):
    target_seq = 2048 if model in ['llama2-13', 'opt'] else 4096
    model_data = [d for d in data if d['model'] == model and d['seq'] == target_seq]
    groups = collections.defaultdict(list)
    for d in model_data:
        groups[d['numcore']].append((d['batch'], d['prepare_time']))
    
    first_label = True
    for numcore, points in groups.items():
        if len(points) < 2:
            continue
        points.sort()
        x, y = zip(*points)
        label = model if first_label else None
        ax.plot(x, y, marker='o', color=color, label=label)
        first_label = False

ax.set_title('Intra-operator search time vs Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Intra-operator search time (s)')
ax.legend()
ax.grid(True)
plt.tight_layout()
output_path = os.path.join(script_dir, 'prepare_time_vs_batch_sensitivity.png')
plt.savefig(output_path)
print(f"Saved {output_path}")


# --- Seq Sensitivity ---
# Create figure for batch=16
target_batch = 16
if True:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model, color in zip(models, colors):
        model_data = [d for d in data if d['model'] == model and d['batch'] == target_batch]
        groups = collections.defaultdict(list)
        for d in model_data:
            groups[d['numcore']].append((d['seq'], d['prepare_time']))
        
        first_label = True
        for numcore, points in groups.items():
            if len(points) < 2:
                continue
            points.sort()
            x, y = zip(*points)
            label = model if first_label else None
            ax.plot(x, y, marker='o', color=color, label=label)
            first_label = False
    
    ax.set_title(f'Intra-operator search time vs Sequence Length (Batch={target_batch})')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Intra-operator search time (s)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    output_path = os.path.join(script_dir, f'prepare_time_vs_seq_batch{target_batch}.png')
    plt.savefig(output_path)
    print(f"Saved {output_path}")

# --- ICBM Time vs Batch Sensitivity ---
fig, ax = plt.subplots(figsize=(12, 8))

for model, color in zip(models, colors):
    target_seq = 2048
    if model not in ['llama2-13', 'opt']:
        target_seq *=2
    model_data = [d for d in data if d['model'] == model and d['seq'] == target_seq]
    groups = collections.defaultdict(list)
    for d in model_data:
        if d['icbm_time'] is not None:
            groups[d['numcore']].append((d['batch'], d['icbm_time']))
    
    first_label = True
    for numcore, points in groups.items():
        if len(points) < 2:
            continue
        points.sort()
        x, y = zip(*points)
        label = model if first_label else None
        ax.plot(x, y, marker='o', color=color, label=label)
        first_label = False

ax.set_title('Inter-operator search time vs Batch Size')
ax.set_xlabel('Batch Size')
ax.set_ylabel('Inter-operator search time (s)')
ax.legend()
ax.grid(True)
plt.tight_layout()
output_path = os.path.join(script_dir, 'icbm_time_vs_batch_sensitivity.png')
plt.savefig(output_path)
print(f"Saved {output_path}")


# --- ICBM Time vs Seq Sensitivity ---
# Create figure for batch=16
target_batch = 16
if True:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model, color in zip(models, colors):
        model_data = [d for d in data if d['model'] == model and d['batch'] == target_batch]
        groups = collections.defaultdict(list)
        for d in model_data:
            if d['icbm_time'] is not None:
                groups[d['numcore']].append((d['seq'], d['icbm_time']))
        
        first_label = True
        for numcore, points in groups.items():
            if len(points) < 2:
                continue
            points.sort()
            x, y = zip(*points)
            label = model if first_label else None
            ax.plot(x, y, marker='o', color=color, label=label)
            first_label = False
    
    ax.set_title(f'Inter-operator search time vs Sequence Length (Batch={target_batch})')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Inter-operator search time (s)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    output_path = os.path.join(script_dir, f'icbm_time_vs_seq_batch{target_batch}.png')
    plt.savefig(output_path)
    print(f"Saved {output_path}")
