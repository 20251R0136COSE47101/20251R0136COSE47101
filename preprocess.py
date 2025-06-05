

import os
from Mediapipe.ApexFrame import find_onset_apex_frames
from AU.AUExtraction import extract_au
import pandas as pd
import cv2

def extract_frames(video_path, save_dir):
    output_folder = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0])
    # print(video_path, save_dir, output_folder)
    # 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # 프레임 저장 (예: frame_00001.jpg)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"총 {frame_count}개의 프레임을 저장했습니다.")
    return output_folder

def preprocess(dataset_root="data/kaggle_2025", \
    onset_output_dir="data/onset_output", apex_output_dir="data/apex_output", \
    AU_output_dir="data/AU_output", dataset_type = "kaggle"):

    if dataset_type == "kaggle":
        lie_dir = os.path.join(dataset_root, "Lie")
        truth_dir = os.path.join(dataset_root, "Truth")
        # print(f"lie_dir: {lie_dir}, truth_dir: {truth_dir}")
        for person in os.listdir(lie_dir):
            p_dir = os.path.join(lie_dir, person)
            for question in os.listdir(p_dir):
                q_dir = os.path.join(p_dir, question)
                find_onset_apex_frames(q_dir, onset_output_dir, apex_output_dir)


        for person in os.listdir(truth_dir):
            p_dir = os.path.join(truth_dir, person)
            for question in os.listdir(p_dir):
                q_dir = os.path.join(p_dir, question)
                find_onset_apex_frames(q_dir, onset_output_dir, apex_output_dir)
        
        
        extract_au(apex_output_dir, AU_output_dir)
        
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
    elif dataset_type == "reallife":
        lie_dir = os.path.join(dataset_root, "Deceptive")
        truth_dir = os.path.join(dataset_root, "Truthful")
        lie_frames_dir = os.path.join(dataset_root, "Deceptive_frames")
        truth_frames_dir = os.path.join(dataset_root, "Truthful_frames")

        for video in os.listdir(lie_dir):
            video_path = os.path.join(lie_dir, video)
            find_onset_apex_frames(extract_frames(video_path, lie_frames_dir), onset_output_dir, apex_output_dir)
        
        for video in os.listdir(truth_dir):
            video_path = os.path.join(truth_dir, video)
            find_onset_apex_frames(extract_frames(video_path, truth_frames_dir), onset_output_dir, apex_output_dir)
        
        extract_au(apex_output_dir, AU_output_dir)

if __name__ == "__main__":
    kaggle_dataset_root = "data\\kaggle_2025"
    reallife_dataset_root = "data\\reallife_2016"
    onset_output = "data\\onset_output"
    apex_output = "data\\apex_output"
    AU_output = "data\\AU_output"
    label_path = "data\\label\\labels.txt"
    preprocess(reallife_dataset_root, onset_output, apex_output, AU_output, "reallife")