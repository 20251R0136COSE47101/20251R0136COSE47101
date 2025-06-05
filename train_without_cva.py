import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np

import os
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

class Trainer:
    def __init__(
        self,
        fpf_model, 
        vertical_model, 
        classification_model,
        dataloader,
        checkpoint: None,
        max_epochs: int = 1,
        lr: float = 5e-4,
        patience: int = 3,
        device: str = f"cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(self.device)
        
        self.fpf_model = fpf_model.to(self.device)
        self.vertical_model = vertical_model.to(self.device)
        self.classification_model = classification_model.to(self.device)
        
        self.dataloader = dataloader
        self.train_dataloader = dataloader.train_dataloader
        self.val_dataloader = dataloader.val_dataloader
        self.test_dataloader = dataloader.test_dataloader

        self.optimizer = optim.Adam(
            list(self.classification_model.parameters()) +
            list(self.fpf_model.parameters()),
            lr=lr
        )
        self.optimizer = optim.Adam(self.classification_model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.max_epochs = max_epochs
        
        self.checkpoint = checkpoint
        
        self.logs = []

    def preprocess_vertical_image(self, img_apex_pil, img_onset_pil, image_size=(224, 224)):    
        try:
            normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
            with torch.no_grad():
                difference_tensor = img_apex_pil - img_onset_pil
                normalized_difference_tensor = normalize_transform(difference_tensor)
                batch_tensor = normalized_difference_tensor.unsqueeze(0) # (C, H, W) -> (1, C, H, W)
            return batch_tensor
        except:
            print("ERROR: NO img!")
    
    def plot_confusion_matrix(self, confusion_matrix, title='Confusion Matrix'):
        plt.figure(figsize=(6, 5))
        sns.set_theme(font_scale=1.2)  # 폰트 크기 조정

        # heatmap 그리기
        ax = sns.heatmap(
            confusion_matrix.numpy() if hasattr(confusion_matrix, "numpy") else confusion_matrix,
            annot=True, fmt='d', cmap='Blues',
            cbar=True, square=True,
            linewidths=0.5, linecolor='gray',
            xticklabels=[0, 1], yticklabels=[0, 1],
            vmin=0, vmax=len(self.test_dataloader.dataset)  # 최대값을 데이터셋 크기로 설정
        )

        ax.set_xlabel("Predicted", fontsize=13, labelpad=10)
        ax.set_ylabel("True", fontsize=13, labelpad=10)
        ax.set_xlabel("Predicted class", fontsize=13, labelpad=10)
        ax.set_ylabel("Actual class", fontsize=13, labelpad=10)
        ax.set_title(title, fontsize=15, pad=12)

        # tick label 크기 및 위치 조정
        ax.xaxis.set_ticklabels(['Positive', 'Negative'], fontsize=12)
        ax.yaxis.set_ticklabels(['True', 'False'], fontsize=12, rotation=0)
        ax.yaxis.set_ticklabels(['Positive', 'Negative'], fontsize=12, rotation=0)

        plt.tight_layout()
        plt.show()
        
    def plot_roc_curve(self, fpr, tpr, auc, title='ROC Curve'):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.show()
        
    def fit(self):
        for epoch in range(1, self.max_epochs + 1):
            # ——— TRAIN STEP ———
            self.fpf_model.train()
            self.vertical_model.train()
            self.classification_model.train()
            train_total_loss = 0.0
            train_correct, train_total = 0, 0

            for onset_img, apex_img, au, labels in tqdm(self.train_dataloader, desc=f"[Train] Epoch {epoch}"):
                onset_img, apex_img, au, labels = onset_img.to(self.device), apex_img.to(self.device), au.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                # overfitting 고려하기. 적은 데이터셋으로 특징 추출 레이어까지 학습 할 것인가?
                fpf_features_onset = self.fpf_model(onset_img)
                fpf_features_apex = self.fpf_model(apex_img)
                fpf_features = fpf_features_onset + fpf_features_apex  # 두 이미지의 특징을 합침
                fpf_features = fpf_features.view(fpf_features.size(0), -1) 
                with torch.no_grad():
                    vertical_features = self.vertical_model(self.preprocess_vertical_image(apex_img, onset_img).squeeze(0))
                    vertical_features = vertical_features.view(vertical_features.size(0), -1)
                # print((vertical_features.shape, fpf_features.shape, au.shape))
                combined_features = torch.cat([fpf_features, vertical_features, au], dim=1)
                
                outputs = self.classification_model(combined_features)
                
                loss = self.criterion(outputs, labels)                
                loss.backward()
                self.optimizer.step()

                train_total_loss += loss.item()
                # preds = outputs.argmax(dim=-1)
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                # print(f" pred: {preds}, labels: {labels}, (preds == labels): {(preds == labels)} (preds == labels).sum().item(): {(preds == labels).sum().item()}")
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_total_loss / len(self.train_dataloader)
            train_acc = train_correct / train_total
            # self.scheduler.step()
            print(f"Epoch {epoch} ▶ train_loss: {avg_train_loss:.4f}, train_acc: {train_acc:.4f}")
            
            # ——— EVAL STEP ———
            self.fpf_model.eval()
            self.vertical_model.eval()
            self.classification_model.eval()
            val_total_loss = 0.0
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for onset_img, apex_img, au, labels in tqdm(self.val_dataloader, desc=f"[Val] Epoch {epoch}"):
                    onset_img, apex_img, au, labels = onset_img.to(self.device), apex_img.to(self.device), au.to(self.device), labels.to(self.device)
                    
                    fpf_features_onset = self.fpf_model(onset_img)
                    fpf_features_apex = self.fpf_model(apex_img)
                    fpf_features = fpf_features_onset + fpf_features_apex
                    vertical_features = self.vertical_model(apex_img)
                    fpf_features = fpf_features.view(fpf_features.size(0), -1) 
                    vertical_features = vertical_features.view(vertical_features.size(0), -1)
                    combined_features = torch.cat([fpf_features, vertical_features, au], dim=1)
                    
                    
                    outputs = self.classification_model(combined_features)
                    loss = self.criterion(outputs, labels.float())
                    val_total_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                    # val_correct += (preds == labels).all(dim=-1).sum().item()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_total_loss / len(self.val_dataloader)
            val_acc = val_correct / val_total
            print(f"Epoch {epoch} ▶ val_loss: {avg_val_loss:.4f}, val_acc: {val_acc:.4f}")

            if (epoch%4==0): 
                torch.save(self.fpf_model.state_dict(), f'./checkpoints/fpf_model_epoch_{epoch}.pth')
                torch.save(self.vertical_model.state_dict(), f'./checkpoints/vertical_model_epoch_{epoch}.pth')
                torch.save(self.classification_model.state_dict(), f'./checkpoints/classification_model_epoch_{epoch}.pth')
        torch.save(self.fpf_model.state_dict(), f'./checkpoints/fpf_model_final.pth')
        torch.save(self.vertical_model.state_dict(), f'./checkpoints/vertical_model_final.pth')
        torch.save(self.classification_model.state_dict(), f'./checkpoints/classification_model_final.pth')
        # ——— 훈련 종료 후 best 모델 로드 ———
        # if self.best_val_acc > 0:
        #     self.model.load_state_dict(torch.load(self.ckpt_path))
        #     print(f"Loaded best model with val_acc = {self.best_val_acc:.4f},epoch = {self.best_epoch}")
            # self.logs.append((self.best_val_acc, self.best_epoch))
            
    def test(self):
        self.fpf_model.eval()
        self.vertical_model.eval()
        self.classification_model.eval()
        if (self.checkpoint != None):
            self.fpf_model.load_state_dict(torch.load(os.path.join(self.checkpoint, "fpf_model_final.pth")))
            self.vertical_model.load_state_dict(torch.load(os.path.join(self.checkpoint, "vertical_model_final.pth")))
            self.classification_model.load_state_dict(torch.load(os.path.join(self.checkpoint, "classification_model_final.pth")))
            
        train_total_loss = 0.0
        correct, total = 0, 0
        TP, FP, FN = 0, 0, 0  # True Positive, False Positive, False Negative
        
        # ROC 계산을 위한 데이터 수집 리스트
        probs_list = []  # 예측 확률
        labels_list = []  # 실제 라벨
        
        with torch.no_grad():
            for onset_img, apex_img, au, labels in tqdm(self.test_dataloader, desc="[Test]"):
                # 데이터 cuda로 보내기
                # inputs = [t.to(self.device) for t in [onset_img, apex_img, au, labels]]
                # onset_img, apex_img, au, labels = inputs
                onset_img, apex_img, au, labels = onset_img.to(self.device), apex_img.to(self.device), au.to(self.device), labels.to(self.device)
                
                
                # 각각 모델 결과값 구하기기
                fpf_features_onset = self.fpf_model(onset_img)
                fpf_features_apex = self.fpf_model(apex_img)
                fpf_features = fpf_features_onset + fpf_features_apex
                vertical_features = self.vertical_model(apex_img)
                fpf_features = fpf_features.view(fpf_features.size(0), -1) 
                vertical_features = vertical_features.view(vertical_features.size(0), -1)
                combined_features = torch.cat([fpf_features, vertical_features, au], dim=1)
                
                # 특징 결합
                combined_features = torch.cat([fpf_features, vertical_features, au], dim=1)
                
                # 예측
                outputs = self.classification_model(combined_features)
                loss = self.criterion(outputs, labels.float())
                
                # 통계 계산
                train_total_loss += loss.item()
                
                # 예측 확률 수집 (ROC 계산용)
                probs = torch.sigmoid(outputs)
                probs_list.append(probs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
            
                #BCEWithLogitsLoss는 손실(loss) 계산 시 sigmoid내장됨. output은 없으니까 sigmoid따로 적용
                # preds = (torch.sigmoid(outputs) > 0.5).float()
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                correct += (preds == labels).sum().item()
                # correct += (preds == labels).all(dim=1).sum().item()
                total += labels.size(0)
                
                # Confusion Matrix 구성 요소 계산
                TP += ((preds == 1) & (labels == 1)).sum().item()
                FP += ((preds == 1) & (labels == 0)).sum().item()
                FN += ((preds == 0) & (labels == 1)).sum().item()
                
        # ROC 데이터 통합
        all_probs = np.concatenate(probs_list)
        all_labels = np.concatenate(labels_list)
        
        # ROC 곡선 및 AUC 계산
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs)
        roc_auc = metrics.roc_auc_score(all_labels, all_probs)
        
        avg_loss = train_total_loss / len(self.test_dataloader)
        accuracy = correct / total
        TN = total - (TP + FP + FN)  # True Negative 계산
    
        # Precision, Recall, F1 계산
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        confusion_matrix = torch.tensor([[TP, FN], [FP, TN]])
        
        self.plot_confusion_matrix(confusion_matrix)
        self.plot_roc_curve(fpr, tpr, roc_auc)
        return train_total_loss/len(self.test_dataloader), correct/total, avg_loss, accuracy, precision, recall, f1, confusion_matrix, roc_auc, fpr, tpr
    
    

    # def log(self):
    #     for epoch, avg_train_loss, val_loss, train_acc, val_acc in self.logs:
    #         print(f"Epoch {epoch} | train_Loss: {avg_train_loss:.4f} | train_Acc: {train_acc:.2%} | val_Loss: {val_loss:.4f} | val_Acc: {val_acc:.2%}")