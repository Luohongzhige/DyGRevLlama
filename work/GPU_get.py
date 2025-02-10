import subprocess, re

used = []

def get_gpu(limit = 0.2):
    output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
    gpu_memory_usages = output.strip().split('\n')
    num_gpus = len(gpu_memory_usages)
    total_memory_output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
    total_memory = total_memory_output.strip().split('\n')
    for i in range(num_gpus):
        gpu_memory_usages[i] = int(gpu_memory_usages[i])
        total_memory[i] = int(total_memory[i])
        gpu_memory_usages[i] = gpu_memory_usages[i] / total_memory[i]
        print(f"GPU {i} memory usage: {gpu_memory_usages[i]}")
        if gpu_memory_usages[i] < limit and i not in used:
            used.append(i)
            return "cuda:" + str(i)

    return None