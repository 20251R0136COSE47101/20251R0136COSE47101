import os
from pathlib import Path

# 윈도우 기준으로 안전한 경로 지정
cache_dir = Path.home() / ".cache" / "torch"
os.environ['TORCH_HOME'] = str(cache_dir)

# 캐시 폴더가 없으면 생성
cache_dir.mkdir(parents=True, exist_ok=True)

# 이후에 모델 로드
from torchvision.models import resnet18, ResNet18_Weights
weights = ResNet18_Weights.IMAGENET1K_V1

model = resnet18(weights=weights)
