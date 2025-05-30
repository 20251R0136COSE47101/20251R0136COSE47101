import os 
import torch
import argparse
from torchvision.transforms import transforms

#Our modules
from preprocess import preprocess
# from inference import inference # optional
from train import Trainer
# from model import Model
from dataloader import DataLoader

from Resnet_18 import resnet_18
from BinaryClassification.classifier import Classifier


#쓸까말까. transfer learning할거면 넣는게 좋을듯
MEAN_RGB = (0.485, 0.456, 0.406)
VAR_RGB = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_RGB, std=VAR_RGB)
])

if __name__ == "__main__":
    #0 parser setting
    parser = argparse.ArgumentParser(description='lie detection model')
    parser.add_argument('--do_preprocess', type=int, default=0, help='1: 전처리 함, 0: 안함. 데이터셋 바뀌면 1옵션으로 하면 됨.')
    parser.add_argument('--mode', type=str, default="train_and_test", help='train_and_test/inference')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--hidden_dim', type=int, default=64, help='은닉층 차원')
    parser.add_argument('--Loss', type=str, default="CrossEntropyLoss", help='CrossEntropyLoss/BCELoss')
    parser.add_argument('--path', type=str, default="binary_classifier.pth", help='dir of saved weight')
    args = parser.parse_args()
    
    #1 input, output directory setting
    dataset_train = "data/dataset/train"
    dataset_test = "data/dataset/test"
    onset_output = "data/onset_output"
    apex_output = "data/apex_output"
    AU_output = "data/AU_output"
    label_path = "data/label"
    
    #2 preprocess
    if args.do_preprocess == 1:
        # 경로에 있는거 지우고 하나요? 
        preprocessed_train = preprocess(dataset_train, dataset_test, onset_output, apex_output, AU_output)
    # else: do nothing
    
    #3 Feature extraction & classification
    fpf_model = resnet_18.ResNet.ResNet50_FPF_Features()
    vertical_model = resnet_18.ResNet.ResNet18_Vertical_Features()
    classification_model = Classifier()
    
    if args.mode == "train_and_test":
        # 전처리에서 어떻게 할지에 따라 경로 넣는방식 바꾸면 됨.
        # ex) dataset_train, test폴더 내에 apex, au를 저장할거면 dataset_train, dataset_test만 넣으면 됨
        trainer = Trainer(fpf_model, vertical_model, classification_model, \
            DataLoader(onset_output, apex_output, AU_output, label_path, transform))
        trainer.fit()
        # trainer.log()
    elif args.mode == "inference":
        # optional. infer_dir가 전처리 안된 영상이라면 전처리 함수 돌리기
        # inference(fpf_model, vertical_model, classification_model, infer_dir)
        pass
    elif args.mode == "test":
        trainer = Trainer(fpf_model, vertical_model, classification_model, \
            DataLoader(onset_output, apex_output, AU_output, label_path, transform))
        trainer.test()
    else: print("mode argument is wrong...\ndo type train_and_test or inference")