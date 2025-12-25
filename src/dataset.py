import torch
import numpy as np

# 🏥 데이터 생성 공장 (dataset.py)
def generate_clinical_semg(n_samples=1000):
    X = []
    y = []
    
    for _ in range(n_samples):
        # 0.2초 -> 1초 (샘플링 늘림)
        time = np.linspace(0, 1, 200) 
        label = np.random.randint(0, 2)
        
        # 1. 기본 신호
        base_signal = np.sin(time * 10) * np.exp(-((time-0.5)**2)/0.02)
        
        if label == 1: # Aspiration (비정상 패턴)
            base_signal += np.sin(time * 50) * 0.3 
            
        # 2. 노이즈 추가 (현실감)
        drift = np.linspace(0, np.random.uniform(-0.5, 0.5), 200)
        power_noise = np.sin(time * 60 * 2 * np.pi) * 0.1
        white_noise = np.random.normal(0, 0.1, 200)
        
        final_signal = base_signal + drift + power_noise + white_noise
        
        X.append(final_signal)
        y.append(label)

    return torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y)