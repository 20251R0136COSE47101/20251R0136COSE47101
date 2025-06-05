import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
#쓸까말까. transfer learning할거면 넣는게 좋을듯
MEAN_RGB = (0.485, 0.456, 0.406)
VAR_RGB = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_RGB, std=VAR_RGB)
])


class LieDetectionDataset(Dataset):
    def __init__(self, onset_dir, apex_dir, au_dir, label_path, transform=None):
        self.onset_dir = onset_dir
        self.apex_dir = apex_dir
        self.au_dir = au_dir
        transform_ = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_RGB, std=VAR_RGB)
        ])
        self.transform = transform_

        # onset_dir에 있는 파일 리스트를 기준으로 샘플명 생성 (ex: 0001.jpg)
        # 확장자 jpg랑 png만 허용 -> 바꿔도 됨
        self.samples = [f for f in os.listdir(onset_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # label 파일(ex: csv, txt 등)에서 샘플명:라벨 dict 생성
        self.labels = self._load_labels(label_path)

    def _load_labels(self, label_path):
        # "sample_name label" 형식의 txt 파일을 읽어와 "name": label형식의 딕셔너리 생성
        labels = {}
        with open(label_path, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                labels[name] = float(label)
        return labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx] # '0001.jpg'
        sample_key = os.path.splitext(sample_name)[0]  # '0001.jpg' -> '0001'
        
        onset_path = os.path.join(self.onset_dir, sample_name)
        apex_path = os.path.join(self.apex_dir, sample_name)
        au_csv_path = os.path.join(self.au_dir, f"{sample_key}.csv")
        
        # 이미지 로드
        onset_img = Image.open(onset_path).convert('RGB')
        apex_img = Image.open(apex_path).convert('RGB')

        if self.transform:
            onset_img = self.transform(onset_img)
            apex_img = self.transform(apex_img)
            
        # AU csv 파일에서 두 번째 행(실제 값)만 추출
        au_df = pd.read_csv(au_csv_path)
        au_vector = au_df.iloc[0].values.astype('float32')  # 첫 번째 데이터 행
        au = torch.tensor(au_vector, dtype=torch.float) # 12개의 텐서(AU개수) (1, 12) shape으로 만들어야함. 나중에 concat시 필요

        # label
        label = self.labels[sample_key]

        return onset_img, apex_img, au, label

class LieDetectionDataLoader:
    def __init__(self, onset_dir, apex_dir, au_dir, label_path, \
        batch_size=32, shuffle=True, num_workers=4, transform=None):
        self.dataset = LieDetectionDataset(onset_dir, apex_dir, au_dir, label_path, transform=transform)
        
        self.dataset_size = len(self.dataset)
        if self.dataset_size == 0:
            raise ValueError("Dataset is empty. Please check the directories and label file.")
        self.train_size = int(self.dataset_size * 0.6)
        self.validation_size = int(self.dataset_size * 0.1)
        self.test_size = self.dataset_size - self.train_size - self.validation_size
        
        self.train_ds, self.val_ds, self.test_ds = random_split(self.dataset, [self.train_size, self.validation_size, self.test_size])
        self.train_dataloader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        self.val_dataloader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        self.test_dataloader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    # def get_loader(self):
    #     return self.dataloader