import numpy as np
from kalman_filter import KF 

#1차원에서 위치 추정 시뮬레이션

# 실제 위치를 생성하는 함수
def true_position(t):
    return 0.1 * t**2  # 가속도가 일정한 운동

# 측정값을 시뮬레이션하는 함수
def simulate_measurement(true_pos, noise_std):
    return true_pos + np.random.normal(0, noise_std)

# 파라미터 설정
n = 2  # 상태 변수 수 (위치와 속도)
m = 1  # 측정 변수 수 (위치만 측정)
dt = 0.1  # 시간 간격
total_time = 10  # 총 시뮬레이션 시간
noise_std = 1.0  # 측정 노이즈의 표준편차

# 칼만 필터 초기화
kf = KF(n, m, dt)

# 시뮬레이션 및 추정
times = np.arange(0, total_time, dt)
true_positions = []
measured_positions = []
estimated_positions = []

for t in times:
    # 실제 위치 계산
    true_pos = true_position(t)
    true_positions.append(true_pos)
    
    # 측정값 시뮬레이션
    measured_pos = simulate_measurement(true_pos, noise_std)
    measured_positions.append(measured_pos)
    
    # 칼만 필터로 위치 추정
    estimated_state = kf.Estimation(np.array([measured_pos]))
    estimated_positions.append(estimated_state[0])

# 결과 출력
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(times, true_positions, label='True Position')
plt.plot(times, measured_positions, 'r.', label='Measured Position')
plt.plot(times, estimated_positions, 'g-', label='Estimated Position')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Kalman Filter: 1D Position Estimation')
plt.grid(True)
plt.show()