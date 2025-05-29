import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

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
        self.criterion = nn.BCEWithLogitsLoss()
        self.max_epochs = max_epochs
        
        # self.optimizer = optim.Adam(
        #     list(self.fpf_model.parameters()) +
        #     list(self.vertical_model.parameters()) +
        #     list(self.classification_model.parameters()),
        #     lr=lr
        # )
        
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

            avg_train_loss = total_loss / len(self.dataloader)
            train_acc      = correct_train / total_train
            self.scheduler.step()
            print(f"Epoch {epoch} ▶ train_loss: {avg_train_loss:.4f}, train_acc: {train_acc:.4f}")


        # ——— 훈련 종료 후 best 모델 로드 ———
        if self.best_val_acc > 0:
            self.model.load_state_dict(torch.load(self.ckpt_path))
            print(f"Loaded best model with val_acc = {self.best_val_acc:.4f},epoch = {self.best_epoch}")
            # self.logs.append((self.best_val_acc, self.best_epoch))
            
    def eval(self):
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        TP, FP, FN = 0, 0, 0  # True Positive, False Positive, False Negative
        
        # ROC 계산을 위한 데이터 수집 리스트
        probs_list = []  # 예측 확률
        labels_list = []  # 실제 라벨
        
        with torch.no_grad():
            for onset_img, apex_img, au, labels in self.dataloader:
                # 데이터 cuda로 보내기
                # inputs = [t.to(self.device) for t in [onset_img, apex_img, au, labels]]
                # onset_img, apex_img, au, labels = inputs
                onset_img, apex_img, au, labels = onset_img.to(self.device), apex_img.to(self.device), au.to(self.device), labels.to(self.device)
                
                
                # 각각 모델 결과값 구하기기
                fpf_features_onset = self.fpf_model(onset_img)
                fpf_features_apex = self.fpf_model(apex_img)
                fpf_features = fpf_features_onset + fpf_features_apex
                vertical = self.vertical_model(apex_img)
                
                # 특징 결합
                combined_features = torch.cat([fpf_features, vertical, au], dim=1)
                
                # 예측
                outputs = self.classification_model(combined_features)
                loss = self.criterion(outputs, labels.float())
                
                # 통계 계산
                total_loss += loss.item()
                
                # 예측 확률 수집 (ROC 계산용)
                probs = torch.sigmoid(outputs)
                probs_list.append(probs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
            
                #BCEWithLogitsLoss는 손실(loss) 계산 시 sigmoid내장됨. output은 없으니까 sigmoid따로 적용
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
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
        
        avg_loss = total_loss / len(self.dataloader)
        accuracy = correct / total
        TN = total - (TP + FP + FN)  # True Negative 계산
    
        # Precision, Recall, F1 계산
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        confusion_matrix = torch.tensor([[TN, FP], [FN, TP]])
                
        return total_loss/len(self.dataloader), correct/total, avg_loss, accuracy, precision, recall, f1, confusion_matrix, roc_auc, fpr, tpr
    
    def plot_confusion_matrix(confusion_matrix, title='Confusion Matrix'):
        plt.figure(figsize=(5, 5))
        sns.heatmap(
            confusion_matrix.numpy(), 
            annot=True, fmt='d', 
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def log(self):
        for epoch, avg_train_loss, val_loss, train_acc, val_acc in self.logs:
            print(f"Epoch {epoch} | train_Loss: {avg_train_loss:.4f} | train_Acc: {train_acc:.2%} | val_Loss: {val_loss:.4f} | val_Acc: {val_acc:.2%}")
