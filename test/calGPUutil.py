import subprocess
import time

def get_gpu_utilization():
    """使用nvidia-smi获取GPU的实时利用率"""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            encoding="utf-8"
        )
        utilization = int(result.strip())
        return utilization
    except subprocess.CalledProcessError as e:
        print(f"Error fetching GPU utilization: {e}")
        return None

def calculate_average_utilization(duration, interval):
    """
    计算在指定时长内的平均GPU利用率
    :param duration: 采样的总时长（秒）
    :param interval: 采样的间隔（秒）
    :return: 平均GPU利用率
    """
    utilization_samples = []
    start_time = time.time()

    while (time.time() - start_time) < duration:
        utilization = get_gpu_utilization()
        if utilization is not None:
            utilization_samples.append(utilization)
            print(f"GPU Utilization: {utilization}%")
        time.sleep(interval)
    print(utilization_samples)

if __name__ == "__main__":
    # 设定采样时长和间隔
    duration = 20  # 例如，采样60秒
    interval = 0.1   # 例如，每5秒采样一次

    average_utilization = calculate_average_utilization(duration, interval)
    print('sucess')