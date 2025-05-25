import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

class LieDetectionDataset(Dataset):
    def __init__(self, onset_dir, apex_dir, au_dir, label_path, transform=None):
        self.onset_dir = onset_dir
        self.apex_dir = apex_dir
        self.transform = transform

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
                labels[name] = int(label)
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
        label = self.labels[sample_name]

        return onset_img, apex_img, au, label

class LieDetectionDataLoader:
    def __init__(self, onset_dir, apex_dir, au_dir, label_path, \
        batch_size=32, shuffle=True, num_workers=0, transform=None):
        self.dataset = LieDetectionDataset(onset_dir, apex_dir, au_dir, label_path, transform=transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    # def get_loader(self):
    #     return self.dataloader