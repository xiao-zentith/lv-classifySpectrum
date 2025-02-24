import numpy as np


def add_noise(spectrum, snr_db):
    # 计算信号功率
    signal_power = np.mean(spectrum ** 2)

    # 将分贝转换为线性比例
    snr_linear = 10 ** (snr_db / 10)

    # 计算噪声功率
    noise_power = signal_power / snr_linear

    # 生成高斯噪声
    noise = np.sqrt(noise_power) * np.random.randn(*spectrum.shape)

    # 添加噪声
    noisy_spectrum = spectrum + noise

    return noisy_spectrum