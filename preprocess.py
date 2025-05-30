import os
from Mediapipe_test.ApexFrame_yoonheon import find_onset_apex_frames
from AU.AUExtraction import extract_au
import pandas as pd

def preprocess(dataset_root="data/dataset", \
    onset_output_dir="data/onset_output", apex_output_dir="data/apex_output", \
    AU_output_dir="data/AU_output", dataset_type = "image"):

    if dataset_type == "image":
        # for sample in os.listdir(dataset_root):
        lie_dir = os.path.join(dataset_root, "Lie")
        truth_dir = os.path.join(dataset_root, "Truth")
        # print(f"lie_dir: {lie_dir}, truth_dir: {truth_dir}")
        
        find_onset_apex_frames(lie_dir, onset_output_dir, apex_output_dir)
        extract_au(apex_output_dir, AU_output_dir)
            
        # for person in os.listdir(truth_dir):
        find_onset_apex_frames(truth_dir, onset_output_dir, apex_output_dir)
        extract_au(apex_output_dir, AU_output_dir)
        
        # TODO 이거 같은곳에 저장되면 이름 같아서 덮어쓰기됨. 그거 이름 바꿔 저장하든 다른 폴더에 저장하든 하면 됨
        
        
        
        for csv in os.listdir(AU_output_dir):
            df = pd.read_csv(csv, header = None)
            result = df.iloc[:, 1438:1462]
            result.to_csv(csv + "_revised", index=False)
        # TODO label file만들어야함
        ''' 데이터로더에서 이렇게 받을거니까 이 형식에 맞춰
        def _load_labels(self, label_path):
        # "sample_name label" 형식의 txt 파일을 읽어와 "name": label형식의 딕셔너리 생성
        labels = {}
        with open(label_path, 'r') as f:
            for line in f:
                name, label = line.strip().split()
                labels[name] = int(label)
        return labels
        '''
    else:
        # 데이터셋 받으면 구현
        return