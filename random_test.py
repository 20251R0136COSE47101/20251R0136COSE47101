# random_test.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from train import Trainer

# 1. DataLoader 래퍼 클래스 (train.py 수정 없이 호환성 유지)
class DataloaderWrapper:
    def __init__(self, dataloader):
        self.train_dataloader = dataloader
        self.val_dataloader = dataloader
        self.test_dataloader = dataloader

# 2. 랜덤 데이터셋 클래스
class RandomTestDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.image_shape = (3, 224, 224)
        self.au_dim = 12
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 레이블에 차원 추가 (스칼라 → [1] 텐서)
        label = torch.tensor(
            [np.random.randint(0, 2)],  # [0] 또는 [1]로 생성
            dtype=torch.float32
        )
        return (
            torch.randint(0, 256, self.image_shape, dtype=torch.uint8),
            torch.randint(0, 256, self.image_shape, dtype=torch.uint8),
            torch.rand(self.au_dim, dtype=torch.float32),
            label  # shape: [1]
        )

# 3. 더미 모델 클래스 (차원 일치화)
class DummyFPFModel(torch.nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fc = torch.nn.Linear(64 * 224 * 224, output_dim)  # Flatten 적용

    def forward(self, x):
        x = x.float() / 255.0  # 정규화
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 1D로 평탄화
        return self.fc(x)

# 4. 테스트 실행 함수
def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 데이터로더 초기화 (래퍼 클래스 사용)
    dataset = RandomTestDataset(num_samples=100)
    dataloader = DataloaderWrapper(DataLoader(dataset, batch_size=16, shuffle=True))

    # 더미 모델 생성 (차원 일치)
    fpf_model = DummyFPFModel().to(device)
    vertical_model = DummyFPFModel().to(device)
    classifier = torch.nn.Linear(64 + 64 + 12, 1).to(device)  # 특징 결합 차원 계산

    # 트레이너 초기화
    trainer = Trainer(
        fpf_model=fpf_model,
        vertical_model=vertical_model,
        classification_model=classifier,
        dataloader=dataloader,
        checkpoint=None,
        device=device
    )

    # 테스트 실행
    results = trainer.test()
    
    # 결과 언패킹
    (_, _, avg_loss, accuracy, precision, recall, f1, cm, roc_auc, fpr, tpr) = results

    # 결과 출력
    print(f"\n[Test Results]")
    print(f"- Loss: {avg_loss:.4f}")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision:.4f}")
    print(f"- Recall: {recall:.4f}")
    print(f"- F1 Score: {f1:.4f}")
    print(f"- AUC: {roc_auc:.4f}")

    # 시각화
    Trainer.plot_confusion_matrix(cm, title='Test Confusion Matrix')
    Trainer.plot_roc_curve(fpr, tpr, roc_auc, title='Test ROC Curve')

if __name__ == "__main__":
    try:
        run_test()
    except ImportError as e:
        print(f"Error: {e}")
        print("Install required packages:")
        print("pip install torch torchvision numpy matplotlib seaborn scikit-learn")