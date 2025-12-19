import json
import matplotlib.pyplot as plt
import numpy as np
import os

def process_data():
    with open(os.path.join(os.path.dirname(__file__), 'data_dump.json'), 'r') as f:
        raw_data = json.load(f)
        
    # Filter and normalize suffixes
    valid_suffixes = {
        '': 'e',
        '_a': 'a',
        '_b': 'b',
        '_c': 'c',
        '_d': 'd'
    }
    
    # Group by (seq_len, model, batch, suffix_label)
    grouped = {}
    
    for entry in raw_data:
        raw_suffix = entry.get('suffix', '')
        if raw_suffix not in valid_suffixes:
            continue
            
        suffix_label = valid_suffixes[raw_suffix]
            
        key = (entry['seq_len'], entry['model'], entry['batch'], suffix_label)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(entry)
        
    final_data = []
    
    for key, entries in grouped.items():
        seq_len, model, batch, suffix_label = key
        
        # Find entry with highest GBps for ICBM times
        icbm_entries = [e for e in entries if 'icbm_schedule_time' in e['data'] and 'icbm_ms' in e['data']]
        
        if not icbm_entries:
            continue
            
        best_icbm_entry = max(icbm_entries, key=lambda x: x['gbps'])
        
        icbm_schedule_time = best_icbm_entry['data']['icbm_schedule_time']
        icbm_ms = best_icbm_entry['data']['icbm_ms']
        
        # Find entry with prepare_time
        prepare_time = best_icbm_entry['data'].get('prepare_time')
        if prepare_time is None:
            for e in entries:
                if 'prepare_time' in e['data']:
                    prepare_time = e['data']['prepare_time']
                    break
        if prepare_time is None:
            prepare_time = 0
            
        final_data.append({
            'seq_len': seq_len,
            'model': model,
            'batch': batch,
            'model_batch': f"{model}-{batch}",
            'suffix': suffix_label,
            'prepare_time': prepare_time,
            'icbm_schedule_time': icbm_schedule_time,
            'icbm_ms': icbm_ms,
            'gbps': best_icbm_entry['gbps']
        })
        
    return final_data

import math

def plot_all_in_one(data):
    # Group by (model, batch, seq_len)
    configs = {}
    for d in data:
        key = (d['model'], d['batch'], d['seq_len'])
        if key not in configs:
            configs[key] = []
        configs[key].append(d)
        
    # Define ordered list for specific layout
    # Row 1: 3 gemma, 2 llama70
    # Row 2: 3 llama13, 2 opt
    ordered_configs = [
        ('gemma2', 'b16', 2048),
        ('gemma2', 'b16', 4096),
        ('gemma2', 'b32', 4096),
        ('llama2-70', 'b16', 2048),
        ('llama2-70', 'b16', 4096),
        ('llama2-13', 'b16', 2048),
        ('llama2-13', 'b16', 4096),
        ('llama2-13', 'b32', 4096),
        ('opt-30', 'b16', 2048),
        ('opt-30', 'b16', 4096),
    ]

    sorted_keys = []
    for key in ordered_configs:
        if key in configs:
            entries = configs[key]
            # Only include if at least one entry has non-zero prepare_time
            if any(e['prepare_time'] > 0 for e in entries):
                sorted_keys.append(key)
            
    n_plots = len(sorted_keys)
    
    # Determine grid size
    cols = 5
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n_plots > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
        
    suffixes_order = ['a', 'b', 'd', 'c', 'e']
    # Define colors for suffixes to be consistent across plots
    colors = {'e': 'black', 'a': 'tab:red', 'b': 'tab:blue', 'c': 'tab:green', 'd': 'tab:purple'}
    
    suffix_display_names = {
        'a': "Original T10",
        'b': "Zero-DP Disabled",
        'd': "Zero-DP Disabled + Limited Shift Dimension",
        'c': "Zero-DP Disabled + Limited Shift Dimension + Fused Activation Size",
        'e': "Zero-DP Disabled + Limited Shift Dimension + Fused Activation Size + Stricter Padding Limit"
    }
    
    for i, key in enumerate(sorted_keys):
        ax = axes_flat[i]
        entries = configs[key]
        model, batch, seq_len = key
        
        # Format title: replace 'b' with 'batch' and add 'seq_len' prefix
        batch_str = batch.replace('b', 'batch')
        ax.set_title(f"{model} - {batch_str} - seq{seq_len}")
        
        # Determine row and column position
        row = i // cols
        col = i % cols
        
        # Only add x-label for bottom row OR if it's the last plot in its column
        if row == rows - 1 or i + cols >= n_plots:
            ax.set_xlabel("Total Compilation Time (s)")
        
        # Only add y-label for leftmost column
        if col == 0:
            ax.set_ylabel("Resulting Execution Time (ms)")
        
        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Collect points to determine limits if needed, but auto-scale usually works
        
        for s in suffixes_order:
            entry = next((e for e in entries if e['suffix'] == s), None)
            if entry:
                x_val = entry['prepare_time'] + entry['icbm_schedule_time']
                y_val = entry['icbm_ms']
                
                ax.scatter(x_val, y_val, c=colors.get(s, 'gray'), s=50)
                # Removed annotation to avoid clutter with long names
                
        ax.grid(True, linestyle='--', alpha=0.6)

    # Hide empty subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].axis('off')
        
    # Add global legend with precise control using multiple legend calls
    from matplotlib.lines import Line2D
    
    # Generate all handles first
    handles_map = {}
    for s in suffixes_order:
        color = colors.get(s, 'gray')
        label = suffix_display_names.get(s, s)
        handle = Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10)
        handles_map[s] = handle
    
    # Create three separate legend groups positioned side-by-side
    # Column 1: Original T10 (a)
    leg1 = fig.legend(handles=[handles_map['a']], 
                     loc='upper center', bbox_to_anchor=(0.2, 0.98), 
                     ncol=1, fontsize=11, frameon=False)
    
    # Column 2: Zero-DP Disabled (b) and Limited Shift (d)
    leg2 = fig.legend(handles=[handles_map['b'], handles_map['d']], 
                     loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                     ncol=1, fontsize=11, frameon=False)
    
    # Column 3: Fused (c) and Stricter (e)
    leg3 = fig.legend(handles=[handles_map['c'], handles_map['e']], 
                     loc='upper center', bbox_to_anchor=(0.8, 0.98), 
                     ncol=1, fontsize=11, frameon=False)
    
    # Add the first two legends back to the figure (since each legend() call replaces the previous)
    fig.add_artist(leg1)
    fig.add_artist(leg2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    output_path = os.path.join(os.path.dirname(__file__), 'huge_comparison_figure.pdf')
    plt.savefig(output_path)
    print(f"Saved huge figure to {output_path}")

if __name__ == "__main__":
    data = process_data()
    plot_all_in_one(data)
