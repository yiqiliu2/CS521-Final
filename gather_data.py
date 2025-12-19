import os
import re
import glob
import json

def parse_log(filepath):
    data = {}
    with open(filepath, 'r') as f:
        content = f.read()
        
    prepare_match = re.search(r'prepare time:\s+([\d\.]+)\s+sec', content)
    if prepare_match:
        data['prepare_time'] = float(prepare_match.group(1))
        
    schedule_match = re.search(r'icbm_ordered schedule time:\s+([\d\.]+)\s+sec', content)
    if schedule_match:
        data['icbm_schedule_time'] = float(schedule_match.group(1))
        
    ms_match = re.search(r'icbm_ordered_ms:\s+([\d\.]+)', content)
    if ms_match:
        data['icbm_ms'] = float(ms_match.group(1))
        
    return data

def scan_folders():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Match folders like outputs_icbm_2048, outputs_icbm_4096, outputs_icbm_2048_a, etc.
    folders = glob.glob(os.path.join(base_dir, 'outputs_icbm_*'))
    
    results = []
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        
        # Extract seq_len
        # Assuming format outputs_icbm_<seq_len>[_suffix]
        match = re.match(r'outputs_icbm_(\d+)(_.*)?', folder_name)
        if not match:
            continue
        seq_len = int(match.group(1))
        suffix = match.group(2) if match.group(2) else ""
        
        # We are looking for logs inside core folders
        # The user mentioned 5888cores, but let's look at all core folders
        core_folders = glob.glob(os.path.join(folder, '*cores'))
        
        for core_folder in core_folders:
            core_count = os.path.basename(core_folder)
            
            # Find .log files
            log_files = glob.glob(os.path.join(core_folder, '*.log'))
            
            for log_file in log_files:
                filename = os.path.basename(log_file)
                # Expected format: <model>-<batch>-<GBps>GBps.log
                # e.g. gemma2-b16-16000GBps.log
                # or maybe just <model>-<batch>.log?
                
                # Regex to parse filename
                # model-batch could be complex, e.g. llama2-13-b32
                # GBps part is usually at the end
                
                file_match = re.match(r'(.+)-(\d+GBps)\.log', filename)
                if file_match:
                    model_batch = file_match.group(1)
                    gbps_str = file_match.group(2)
                    gbps = int(gbps_str.replace('GBps', ''))
                else:
                    # Maybe no GBps?
                    # But user said "use the log with the highest GBps"
                    # Let's assume logs of interest have GBps
                    continue
                
                # Parse model and batch from model_batch
                # e.g. gemma2-b16 -> model=gemma2, batch=b16
                # llama2-13-b32 -> model=llama2-13, batch=b32
                # It seems batch is always at the end of model_batch
                last_dash = model_batch.rfind('-')
                if last_dash != -1:
                    model = model_batch[:last_dash]
                    batch = model_batch[last_dash+1:]
                else:
                    model = model_batch
                    batch = "unknown"

                log_data = parse_log(log_file)
                
                results.append({
                    'folder': folder_name,
                    'seq_len': seq_len,
                    'suffix': suffix,
                    'core_count': core_count,
                    'model': model,
                    'batch': batch,
                    'gbps': gbps,
                    'log_file': log_file,
                    'data': log_data
                })
                
    return results

if __name__ == "__main__":
    data = scan_folders()
    print(json.dumps(data, indent=2))
