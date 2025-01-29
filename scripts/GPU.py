import subprocess
import time

def get_gpu_usage():
    # 使用nvidia-smi命令来获取GPU利用率
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', '0'], 
                            stdout=subprocess.PIPE, 
                            text=True)
    return result.stdout.strip()

def log_gpu_usage(duration=10):
    list = []
    start_time = time.time()
    while time.time() - start_time < duration:
        gpu_usage = get_gpu_usage()
        list.append(gpu_usage)
        print(f"GPU0 utilization: {gpu_usage}%")
        time.sleep(0.1)
    print(list)

# 运行函数，持续1分钟获取GPU0的利用率
log_gpu_usage()
