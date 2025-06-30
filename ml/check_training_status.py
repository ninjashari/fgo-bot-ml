#!/usr/bin/env python3
"""
Training Status Checker

Monitor the progress of ongoing FGO image classification training.
"""

import os
import json
import glob
import time
from pathlib import Path
import subprocess


def check_gpu_usage():
    """Check current GPU usage."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
            return {
                'gpu_utilization': int(gpu_util),
                'memory_used': int(mem_used),
                'memory_total': int(mem_total),
                'temperature': int(temp)
            }
    except:
        pass
    return None


def find_training_outputs():
    """Find all training output directories."""
    pattern = "training_output_*"
    outputs = glob.glob(pattern)
    return sorted(outputs, key=lambda x: os.path.getmtime(x), reverse=True)


def get_latest_training_status():
    """Get status of the most recent training."""
    outputs = find_training_outputs()
    if not outputs:
        return None
    
    latest_output = outputs[0]
    
    # Check for training results
    results_file = os.path.join(latest_output, "training_results.json")
    config_file = os.path.join(latest_output, "training_config.json")
    log_file = os.path.join(latest_output, "logs", "training.log")
    
    status = {
        'output_dir': latest_output,
        'is_complete': os.path.exists(results_file),
        'config_exists': os.path.exists(config_file),
        'log_exists': os.path.exists(log_file)
    }
    
    # Load config if available
    if status['config_exists']:
        try:
            with open(config_file, 'r') as f:
                status['config'] = json.load(f)
        except:
            status['config'] = None
    
    # Load results if complete
    if status['is_complete']:
        try:
            with open(results_file, 'r') as f:
                status['results'] = json.load(f)
        except:
            status['results'] = None
    
    # Get last few log lines
    if status['log_exists']:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                status['last_log_lines'] = lines[-10:] if len(lines) > 10 else lines
        except:
            status['last_log_lines'] = []
    
    # Check for model files
    models_dir = os.path.join(latest_output, "models")
    if os.path.exists(models_dir):
        model_files = os.listdir(models_dir)
        status['model_files'] = model_files
        status['has_best_model'] = 'best_model.pth' in model_files
    
    return status


def format_time_elapsed(start_time_str):
    """Format elapsed time from start time string."""
    try:
        # Parse timestamp from log
        # Format: 2024-12-30 13:45:35,123
        time_str = start_time_str.split(' - ')[0]
        start_time = time.strptime(time_str.split(',')[0], "%Y-%m-%d %H:%M:%S")
        start_timestamp = time.mktime(start_time)
        
        elapsed = time.time() - start_timestamp
        
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except:
        return "Unknown"


def main():
    """Main status checker."""
    print("🔍 FGO Training Status Checker")
    print("=" * 50)
    
    # Check GPU status
    gpu_status = check_gpu_usage()
    if gpu_status:
        print(f"🔥 GPU Status:")
        print(f"   Utilization: {gpu_status['gpu_utilization']}%")
        print(f"   Memory: {gpu_status['memory_used']}/{gpu_status['memory_total']} MB")
        print(f"   Temperature: {gpu_status['temperature']}°C")
        
        # Determine if training is likely running
        if gpu_status['gpu_utilization'] > 50:
            print("   🟢 Training appears to be ACTIVE")
        else:
            print("   🟡 GPU usage low - training may be idle/complete")
    else:
        print("❌ Unable to check GPU status")
    
    print()
    
    # Check training outputs
    training_status = get_latest_training_status()
    if not training_status:
        print("❌ No training outputs found")
        return
    
    print(f"📁 Latest Training: {training_status['output_dir']}")
    
    if training_status['config']:
        config = training_status['config']
        print(f"📊 Configuration:")
        print(f"   Model: {config.get('model_name', 'Unknown')}")
        print(f"   Dataset: {config.get('dataset_type', 'Unknown')}")
        print(f"   Epochs: {config.get('num_epochs', 'Unknown')}")
        print(f"   Batch Size: {config.get('batch_size', 'Unknown')}")
    
    if training_status['is_complete']:
        print("✅ Training Status: COMPLETED")
        
        if training_status['results']:
            results = training_status['results']
            print(f"🏆 Final Results:")
            print(f"   Best Val Accuracy: {results.get('best_val_accuracy', 'N/A'):.2f}%")
            print(f"   Test Accuracy: {results.get('test_accuracy', 'N/A'):.2f}%")
            print(f"   Training Time: {results.get('total_training_time', 0)/3600:.1f}h")
    else:
        print("🏃 Training Status: IN PROGRESS")
        
        # Try to extract progress from logs
        if training_status['last_log_lines']:
            recent_lines = [line.strip() for line in training_status['last_log_lines'][-5:]]
            print(f"📝 Recent Progress:")
            for line in recent_lines:
                if line:
                    print(f"   {line}")
    
    # Model files status
    if training_status.get('model_files'):
        print(f"💾 Model Files: {len(training_status['model_files'])} files")
        if training_status['has_best_model']:
            print("   ✅ Best model available")
    
    print()
    print("💡 Monitoring Commands:")
    print(f"   GPU: watch -n 2 nvidia-smi")
    print(f"   Logs: tail -f {training_status['output_dir']}/logs/training.log")
    print(f"   TensorBoard: tensorboard --logdir {training_status['output_dir']}/tensorboard")


if __name__ == "__main__":
    main() 