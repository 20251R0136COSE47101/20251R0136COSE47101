import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        fpf_model, 
        vertical_model, 
        classification_model,
        dataloader,
        lr: float = 5e-4,
        max_epochs: int = 9,
        patience: int = 3,
        device: str = f"cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.fpf_model = fpf_model.to(self.device)
        self.vertical_model = vertical_model.to(self.device)
        self.classification_model = classification_model.to(self.device)
        self.dataloader = dataloader

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.max_epochs = max_epochs
        
        self.logs = []

        

    def fit(self):
        for epoch in range(1, self.max_epochs + 1):
            # ——— TRAIN STEP ———
            self.model.train()
            total_loss = 0.0
            correct_train, total_train = 0, 0

            for onset_img, apex_img, au, labels in tqdm(self.dataloader, desc=f"[Train] Epoch {epoch}"):
                onset_img, apex_img, au, labels = onset_img.to(self.device), apex_img.to(self.device), au.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                # overfitting 고려하기. 적은 데이터셋으로 특징 추출 레이어까지 학습 할 것인가?
                with torch.no_grad():
                    fpf_features_onset = self.fpf_model(onset_img)
                    fpf_features_apex = self.fpf_model(apex_img)
                    fpf_features = fpf_features_onset + fpf_features_apex  # 두 이미지의 특징을 합침
                    vertical_features = self.vertical_model(apex_img)
                combined_features = torch.cat([fpf_features, vertical_features, au], dim=1)
                
                outputs = self.classification_model(combined_features)
                
                loss = self.criterion(outputs, labels)                
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            avg_train_loss = total_loss / len(self.train_loader)
            train_acc      = correct_train / total_train
            self.scheduler.step()
            print(f"Epoch {epoch} ▶ train_loss: {avg_train_loss:.4f}, train_acc: {train_acc:.4f}")


        # ——— 훈련 종료 후 best 모델 로드 ———
        if self.best_val_acc > 0:
            self.model.load_state_dict(torch.load(self.ckpt_path))
            print(f"Loaded best model with val_acc = {self.best_val_acc:.4f},epoch = {self.best_epoch}")
            # self.logs.append((self.best_val_acc, self.best_epoch))
            

    def log(self):
        for epoch, avg_train_loss, val_loss, train_acc, val_acc in self.logs:
            print(f"Epoch {epoch} | train_Loss: {avg_train_loss:.4f} | train_Acc: {train_acc:.2%} | val_Loss: {val_loss:.4f} | val_Acc: {val_acc:.2%}")
