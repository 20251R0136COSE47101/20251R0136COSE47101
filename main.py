import os 
import torch
import argparse
from torchvision.transforms import transforms

#Our modules
from preprocess import preprocess
# from inference import inference # optional
from train import Trainer
from dataloader import LieDetectionDataLoader
from Resnet_18 import resnet_18
from BinaryClassification.classifier import Classifier


if __name__ == "__main__":
    #0 parser setting
    parser = argparse.ArgumentParser(description='lie detection model')
    parser.add_argument('--do_preprocess', type=int, default=0, help='1: 전처리 함, 0: 안함. 데이터셋 바뀌면 1옵션으로 하면 됨.')
    parser.add_argument('--mode', type=str, default="train_and_val", help='train_and_val/inference/test')
    parser.add_argument('--epoch', type=int, default=5, help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.0001, help='학습률')
    parser.add_argument('--hidden_dim', type=int, default=256, help='은닉층 차원')
    parser.add_argument('--Loss', type=str, default="CrossEntropyLoss", help='CrossEntropyLoss/BCELoss')
    parser.add_argument('--path', type=str, default="binary_classifier.pth", help='dir of saved weight')
    args = parser.parse_args()
    
    #1 input, output directory setting
    kaggle_dataset_root = "data\\kaggle_2025"
    reallife_dataset_root = "data\\reallife_2016"
    onset_output = "data\\onset_output"
    apex_output = "data\\apex_output"
    AU_output = "data\\AU_output"
    label_path = "data\\label\\labels.txt"
    checkpoint_path = "checkpoints\\"
    

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    #2 preprocess
    if args.do_preprocess == 1:
        # 경로에 있는거 지우고 하나요? 
        # preprocess(kaggle_dataset_root, onset_output, apex_output, AU_output, "kaggle")
        preprocess(reallife_dataset_root, onset_output, apex_output, AU_output, "reallife")
    
    #3 Feature extraction & classification
    fpf_model = resnet_18.ResNet.ResNet18_FPF_Features()
    vertical_model = resnet_18.ResNet.ResNet18_Vertical_Features()
    classification_model = Classifier()
    
    if args.mode == "train_and_val":
        # 전처리에서 어떻게 할지에 따라 경로 넣는방식 바꾸면 됨.
        # ex) dataset_train, test폴더 내에 apex, au를 저장할거면 dataset_train, dataset_test만 넣으면 됨
        loader = LieDetectionDataLoader(onset_output, apex_output, AU_output, label_path)
        trainer = Trainer(fpf_model, vertical_model, classification_model, \
            LieDetectionDataLoader(onset_output, apex_output, AU_output, label_path), checkpoint_path, args.epoch)
        trainer.fit()
        # trainer.log()
    elif args.mode == "inference":
        # optional. infer_dir가 전처리 안된 영상이라면 전처리 함수 돌리기
        # inference(fpf_model, vertical_model, classification_model, infer_dir)
        pass
    elif args.mode == "test":
        trainer = Trainer(fpf_model, vertical_model, classification_model, \
            LieDetectionDataLoader(onset_output, apex_output, AU_output, label_path), checkpoint_path)
        trainer.test()
    else:
        print("mode argument is wrong...\ndo type train_and_test or inference")