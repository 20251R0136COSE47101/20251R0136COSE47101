import os
from Mediapipe_test.ApexFrame_yoonheon import find_onset_apex_frames
from AU.AUExtraction import extract_au
import pandas as pd

def preprocess(dataset_root="data/dataset", \
    onset_output_dir="data/onset_output", apex_output_dir="data/apex_output", \
    AU_output_dir="data/AU_output", dataset_type = "image"):

    if dataset_type == "image":
        for sample in os.listdir(dataset_root):
            lie_dir = os.path.join(dataset_root, "Lie")
            truth_dir = os.path.join(dataset_root, "Truth")
            for person in os.listdir(truth_dir):
                q_dir = os.path.join(truth_dir, person)
                for q in os.listdir(q_dir):
                    final_dir = os.path.join(q_dir, q)
                    find_onset_apex_frames(final_dir, onset_output_dir, apex_output_dir)
                    extract_au(apex_output_dir, AU_output_dir)
            
            for person in os.listdir(lie_dir):
                q_dir = os.path.join(lie_dir, person)
                for q in os.listdir(q_dir):
                    final_dir = os.path.join(q_dir, q)
                    find_onset_apex_frames(final_dir, onset_output_dir, apex_output_dir)
                    extract_au(apex_output_dir, AU_output_dir)
        
        for csv in os.listdir(AU_output_dir):
            df = pd.read_csv(csv, header = None)
            result = df.iloc[:, 1438:1462]
            result.to_csv(csv + "_revised", index=False)
            
    else:
        # 데이터셋 받으면 구현
        return