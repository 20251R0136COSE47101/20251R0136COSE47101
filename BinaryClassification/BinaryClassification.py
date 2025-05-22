import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt

from resnet_18 import preprocess_vertical_image, ResNet, preprocess_image

class BinaryClassificationDataset(Dataset):
    def __init__(self, num_samples=1000, Feature_dim = 708*14*14, AU_dim = 0, random_seed=42):
        # 재현성을 위한 시드 설정
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 무작위 데이터 생성
        self.Feature = torch.randn(num_samples, 708, 14, 14)
        self.AU = torch.randint(0, 2, (num_samples, AU_dim)).float()
        self.AU_dim = AU_dim
        
        # 단순한 규칙 기반 레이블 생성 (첫 번째 값이 0보다 크면 1, 아니면 0)
        self.y = torch.zeros(num_samples)
        self.y = (self.Feature[:, 0, 0, 0] > 0).long()
        
    def __len__(self):
        return len(self.Feature)
    
    def __getitem__(self, idx):
        feature_flat = self.Feature[idx].flatten()
        if self.AU_dim != 0:
            concated = torch.cat([feature_flat, self.AU[idx]], dim=0)
        else:
            concated = feature_flat
        return concated, self.y[idx]

# 이진 분류 모델 정의
class BinaryClassifier(nn.Module):
    def __init__(self, Feature_dim = 708*14*14, AU_dim = 0, hidden_dim=256):
        super(BinaryClassifier, self).__init__()
        self.flatten = nn.Flatten()
        concat_dim = Feature_dim + AU_dim # 138,768 + AU
        self.layer1 = nn.Linear(concat_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        # print(x.shape)
        # print(x)
        # print(x.squeeze())
        return x.squeeze()

def train_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(device)
    
    # 데이터셋
    train_dataset = BinaryClassificationDataset(num_samples=800)
    test_dataset = BinaryClassificationDataset(num_samples=200, random_seed=43)
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    
    model = BinaryClassifier(Feature_dim = 708*14*14, AU_dim = 0, hidden_dim=args.hidden_dim)
    # model.to(device)
    Loss = {"BCELoss": nn.BCELoss(), "CrossEntropyLoss": nn.CrossEntropyLoss()}
    criterion = Loss[args.Loss]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 학습 기록용 리스트
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(args.epochs):
        # train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # inputs.to(device)
            # labels.to(device)
            optimizer.zero_grad()
            
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            # backward
            loss.backward()
            optimizer.step()
            
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 에폭 평균 손실 및 정확도 계산
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 평가 모드
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                
                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 테스트 평균 손실 및 정확도 계산
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 진행상황 출력
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                 f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # 학습 결과 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return model

def predict(feature):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    input_tensor = feature.view(feature.size(0), -1)       
    #input_tensor = torch.cat([feature_flat, au], dim=1) 

    model = BinaryClassifier(Feature_dim = 708*14*14, AU_dim = 0, hidden_dim=args.hidden_dim)
    model.load_state_dict(torch.load(args.path, weights_only=True))
    # model.to(device)
    Loss = {"BCELoss": nn.BCELoss(), "CrossEntropyLoss": nn.CrossEntropyLoss()}
    criterion = Loss[args.Loss]

    model.eval()
    test_loss = 0
        
    with torch.no_grad():
        
        outputs = model(input_tensor)
        res = outputs.item()
        # predicted = (outputs > 0.5).float()
        print(f"거짓말 가능성: {res*100:.4f}%")
                

if __name__ == '__main__':
    
    # 명령행 인자 파싱 설정
    parser = argparse.ArgumentParser(description='128차원 데이터 이진 분류 모델')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--hidden_dim', type=int, default=64, help='은닉층 차원')
    parser.add_argument('--Loss', type=str, default="CrossEntropyLoss", help='CrossEntropyLoss/BCELoss')
    parser.add_argument('--mode', type=str, default="train_and_test", help='train_and_test/predict')
    parser.add_argument('--path', type=str, default="binary_classifier.pth", help='dir of saved weight')
    args = parser.parse_args()
    
    
    ver_input = preprocess_vertical_image("C:\\Users\\woosu\\gitProjects\\20251R0136COSE47101\\Test\\Atul10459.png", 
                                      "C:\\Users\\woosu\\gitProjects\\20251R0136COSE47101\\Test\\Atul03069.png")

    vertical_model = ResNet.ResNet18_Vertical_Features()
    vertical_output = vertical_model(ver_input)


    #이미지 경로를 넣으면 resnet 돌린 data가 FPF_output에 1 196 14 14 로 나옴. 각각 fpf_apex(oneset)_output으로 나오고, 이를 elementwise로 해준 최종 결과가 fpf_output
    fpf_apex_input = preprocess_image("C:\\Users\\woosu\\gitProjects\\20251R0136COSE47101\\Test\\Atul10459.png")
    fpf_onset_input = preprocess_image("C:\\Users\\woosu\\gitProjects\\20251R0136COSE47101\\Test\\Atul03069.png")

    fpf_model = ResNet.ResNet50_FPF_Features()

    fpf_apex_output = fpf_model(fpf_apex_input)
    fpf_onset_output = fpf_model(fpf_onset_input)

    fpf_output = fpf_apex_output + fpf_onset_output


    real_output = torch.cat((vertical_output,fpf_output), dim=1) #classification
    
    if args.mode == "train_and_test":
        model = train_model()
            # 모델 저장
        torch.save(model.state_dict(), 'binary_classifier.pth')
        print("모델이 'binary_classifier.pth'로 저장되었습니다.")
    elif args.mode == "predict":
        predict(real_output)
    
