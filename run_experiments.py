import subprocess
import time

def run_experiment(config_name):
    print(f"\nRunning experiment with {config_name} configuration...")
    subprocess.run(['python', 'train.py', '--config', config_name])

if __name__ == '__main__':
    # Create saved directory if it doesn't exist
    import os
    os.makedirs('saved', exist_ok=True)
    
    # Run all configurations
    configs = ['config', 'overfit', 'underfit', 'optimal']
    start_time = time.time()
    
    for config in configs:
        run_experiment(config)
    
    total_time = time.time() - start_time
    print(f"\nAll experiments completed in {total_time:.2f} seconds")