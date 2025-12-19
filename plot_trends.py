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

# Plot Prepare Time vs NumCore
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each model
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for model, color in zip(models, colors):
    model_data = [d for d in data if d['model'] == model]
    # Group by (batch, seq) to draw lines
    groups = collections.defaultdict(list)
    for d in model_data:
        groups[(d['batch'], d['seq'])].append((d['numcore'], d['prepare_time']))
    
    first_label = True
    for (batch, seq), points in groups.items():
        if len(points) < 2:
            continue
        points.sort()
        x, y = zip(*points)
        label = model if first_label else None
        ax.plot(x, y, marker='o', color=color, label=label)
        first_label = False

ax.set_title('Intra-operator search time vs NumCore (All Models)')
ax.set_xlabel('NumCore')
ax.set_ylabel('Intra-operator search time (s)')
ax.legend()
ax.grid(True)

plt.tight_layout()
output_path = os.path.join(script_dir, 'prepare_time_vs_numcore.png')
plt.savefig(output_path)
print(f"Saved {output_path}")

# Plot ICBM Time vs NumCore
fig, ax = plt.subplots(figsize=(12, 8))

for model, color in zip(models, colors):
    model_data = [d for d in data if d['model'] == model]
    # Group by (batch, seq) to draw lines
    groups = collections.defaultdict(list)
    for d in model_data:
        if d['icbm_time'] is not None:
            groups[(d['batch'], d['seq'])].append((d['numcore'], d['icbm_time']))
    
    first_label = True
    for (batch, seq), points in groups.items():
        if len(points) < 2:
            continue
        points.sort()
        x, y = zip(*points)
        label = model if first_label else None
        ax.plot(x, y, marker='o', color=color, label=label)
        first_label = False

ax.set_title('Inter-operator search time vs NumCore (All Models)')
ax.set_xlabel('NumCore')
ax.set_ylabel('Inter-operator search time (s)')
ax.legend()
ax.grid(True)

plt.tight_layout()
output_path = os.path.join(script_dir, 'icbm_time_vs_numcore.png')
plt.savefig(output_path)
print(f"Saved {output_path}")

# Plot Prepare Time vs Batch and Seq
fig, axes = plt.subplots(len(models), 2, figsize=(15, 5 * len(models)))
if len(models) == 1: axes = [axes]

for i, model in enumerate(models):
    model_data = [d for d in data if d['model'] == model]
    
    # Prepare vs Batch
    ax = axes[i][0] if len(models) > 1 else axes[0]
    batches = [d['batch'] for d in model_data]
    prepares = [d['prepare_time'] for d in model_data]
    ax.scatter(batches, prepares, alpha=0.7)
    ax.set_title(f'{model} - Intra-operator search time vs Batch')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Intra-operator search time (s)')
    ax.grid(True)

    # Prepare vs Seq
    ax = axes[i][1] if len(models) > 1 else axes[1]
    seqs = [d['seq'] for d in model_data]
    prepares = [d['prepare_time'] for d in model_data]
    ax.scatter(seqs, prepares, alpha=0.7, c='orange')
    ax.set_title(f'{model} - Intra-operator search time vs Seq')
    ax.set_xlabel('Seq')
    ax.set_ylabel('Intra-operator search time (s)')
    ax.grid(True)

plt.tight_layout()
output_path = os.path.join(script_dir, 'prepare_time_vs_batch_seq.png')
plt.savefig(output_path)
print(f"Saved {output_path}")

# Calculate and print correlations

print("\nCorrelations:")
print(f"{'Model':<15} {'Metric':<15} {'vs NumCore':<12} {'vs Batch':<12} {'vs Seq':<12}")
print("-" * 70)

for model in models:
    model_data = [d for d in data if d['model'] == model]
    if not model_data:
        continue
    
    numcores = [d['numcore'] for d in model_data]
    batches = [d['batch'] for d in model_data]
    seqs = [d['seq'] for d in model_data]
    prepares = [d['prepare_time'] for d in model_data]
    icbms = [d['icbm_time'] for d in model_data if d['icbm_time'] is not None]
    
    # Prepare Time Correlations
    if len(prepares) > 1:
        corr_core = np.corrcoef(numcores, prepares)[0, 1] if np.std(numcores) > 0 and np.std(prepares) > 0 else 0
        corr_batch = np.corrcoef(batches, prepares)[0, 1] if np.std(batches) > 0 and np.std(prepares) > 0 else 0
        corr_seq = np.corrcoef(seqs, prepares)[0, 1] if np.std(seqs) > 0 and np.std(prepares) > 0 else 0
        print(f"{model:<15} {'Intra-op Time':<15} {corr_core:<12.2f} {corr_batch:<12.2f} {corr_seq:<12.2f}")

    # ICBM Time Correlations
    # Note: icbms list might be shorter if some are None, need to align
    icbm_data = [d for d in model_data if d['icbm_time'] is not None]
    if len(icbm_data) > 1:
        icbm_cores = [d['numcore'] for d in icbm_data]
        icbm_batches = [d['batch'] for d in icbm_data]
        icbm_seqs = [d['seq'] for d in icbm_data]
        icbm_vals = [d['icbm_time'] for d in icbm_data]
        
        corr_core = np.corrcoef(icbm_cores, icbm_vals)[0, 1] if np.std(icbm_cores) > 0 and np.std(icbm_vals) > 0 else 0
        corr_batch = np.corrcoef(icbm_batches, icbm_vals)[0, 1] if np.std(icbm_batches) > 0 and np.std(icbm_vals) > 0 else 0
        corr_seq = np.corrcoef(icbm_seqs, icbm_vals)[0, 1] if np.std(icbm_seqs) > 0 and np.std(icbm_vals) > 0 else 0
        print(f"{model:<15} {'Inter-op Time':<15} {corr_core:<12.2f} {corr_batch:<12.2f} {corr_seq:<12.2f}")
