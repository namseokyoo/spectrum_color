import numpy as np

# 파장 범위 설정 (380-780nm)
wavelengths = np.arange(380, 781, 5)

# R 스펙트럼 (620-700nm 영역에 피크)
r_peak = 650
r_width = 30
r_intensity = np.exp(-0.5 * ((wavelengths - r_peak) / r_width) ** 2)

# G 스펙트럼 (500-570nm 영역에 피크)
g_peak = 530
g_width = 30
g_intensity = np.exp(-0.5 * ((wavelengths - g_peak) / g_width) ** 2)

# B 스펙트럼 (450-490nm 영역에 피크)
b_peak = 470
b_width = 30
b_intensity = np.exp(-0.5 * ((wavelengths - b_peak) / b_width) ** 2)

# 결과 출력
print('R 스펙트럼 예시:')
for w, i in zip(wavelengths, r_intensity):
    print(f'{w} {i:.6f}')

print('\nG 스펙트럼 예시:')
for w, i in zip(wavelengths, g_intensity):
    print(f'{w} {i:.6f}')

print('\nB 스펙트럼 예시:')
for w, i in zip(wavelengths, b_intensity):
    print(f'{w} {i:.6f}')
