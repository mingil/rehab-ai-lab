import torch
import torch.nn as nn
import torch.optim as optim
import argparse  # 👈 프로들의 도구 (터미널에서 옵션 받기)
from model import Hybrid_DeepSwallow  # 방금 만든 부품 가져오기
from dataset import generate_clinical_semg  # (가정: dataset.py에 함수가 있다고 칩시다)

# 임시로 데이터 생성 함수 여기에 포함 (나중에 dataset.py로 옮기셔도 됩니다)
import numpy as np


def main(args):
    # 1. 설정 출력
    print(f"🚀 [DeepSwallow] 학습 시작 | Epochs: {args.epochs} | LR: {args.lr}")

    # 2. 장비 세팅
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Hybrid_DeepSwallow().to(device)

    # 3. 데이터 준비
    X_train, y_train = generate_clinical_semg(1000)
    X_train, y_train = X_train.to(device), y_train.to(device)

    # 4. 학습 도구
    weights = torch.tensor([1.0, 5.0]).to(device)  # 가중치 적용
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5. 루프 돌리기
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.4f}")

    # 6. 모델 저장 (여기가 핵심!)
    torch.save(model.state_dict(), f"models/deepswallow_epoch{args.epochs}.pth")
    print(f"💾 모델 저장 완료: models/deepswallow_epoch{args.epochs}.pth")


if __name__ == "__main__":
    # 터미널에서 받을 옵션 정의
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="반복 횟수")
    parser.add_argument("--lr", type=float, default=0.001, help="학습률")
    args = parser.parse_args()

    main(args)
