import os
import re

base_dir = './benchmark_scripts/outputs_a'

def parse_file(filepath):
    prepare_time = None
    icbm_time = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "prepare time:" in line:
                    # Example: prepare time: 231.76976306992583 sec
                    match = re.search(r'prepare time:\s*([\d\.]+)', line)
                    if match:
                        prepare_time = match.group(1)
                if "icbm_ordered schedule time:" in line:
                    # Example: icbm_ordered schedule time: 9.010738058947027 sec
                    match = re.search(r'icbm_ordered schedule time:\s*([\d\.]+)', line)
                    if match:
                        icbm_time = match.group(1)
    except Exception as e:
        return None, None
    return prepare_time, icbm_time

print(f"{'Model':<15} {'Batch':<8} {'Seq':<8} {'NumCore':<10} {'Prepare Time (s)':<20} {'ICBM Time (s)':<20}")
print("-" * 85)

for model in os.listdir(base_dir):
    model_dir = os.path.join(base_dir, model)
    if not os.path.isdir(model_dir):
        continue
    
    for filename in os.listdir(model_dir):
        if not filename.startswith("test-bw-"):
            continue
        
        # Parse filename: test-bw-{batch}-{seq}-{bw}-{numcore}-{sram}-...
        parts = filename.split('-')
        if len(parts) < 6:
            continue
            
        try:
            batch = parts[2]
            seq = parts[3]
            numcore = parts[5]
        except IndexError:
            continue

        filepath = os.path.join(model_dir, filename)
        prepare_time, icbm_time = parse_file(filepath)
        
        if prepare_time:
            icbm_val = icbm_time if icbm_time else "N/A"
            print(f"{model:<15} {batch:<8} {seq:<8} {numcore:<10} {prepare_time:<20} {icbm_val:<20}")
