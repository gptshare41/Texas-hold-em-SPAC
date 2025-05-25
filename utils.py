import numpy as np

def result_function(state):
    # 간단한 예시: 랜덤 승자 결정 (실제로는 핸드 평가 필요)
    return 'A' if np.random.rand() > 0.5 else 'B'
