"""
둘 중 하나 형식으로 import하면 됨
1. from 폴더이름.파이썬파일이름 import 그 파이썬 파일에서 사용할 함수, class명
2. import 폴더이름.파이썬파일이름 


예시: from AU_extraction.AUExtraction import ~~ or import AU_extraction.AUExtraction
"""
from AU.AUExtraction import extract_au


def preprocess(dataset_train="data/dataset/train", dataset_test="data/dataset/test", \
    onset_output_dir="data/onset_output", apex_output_dir="data/apex_output", \
    AU_output_dir="data/AU_output"):
    # 각 영상을 프레임단위로 잘라 나온 사진들을 한 질문에 대한 답변마다 
    for sample in os.listdir(dataset_train):
        #0. input이 영상이면 여기서 자르기
        
        #1. onset frame저장
        onset_frame = 각 영상자른 데이터셋 폴더에서 onset찾기
        do save onset_frames of dataset_train
        
        #2. apex를 추출해서 apex_output_dir에 저장.
        apex_frame = apex를 뽑아내는 함수
        do save apex_frames of dataset_train
        
        #3. AU를 추출해서 AU_output_dir에 저장.
        AU = extract_au
        AU(apex_output_dir, AU_output_dir)
        #do save AUs of dataset_train
        
        #4. label 정보 담긴 파일(.txt로) 생성
        # sample_name label 형식으로 2열로 만들기
        
    for sample in os.listdir(dataset_train):
        #0. input이 영상이면 여기서 자르기
        
        #1. onset frame저장
        onset_frame = 각 영상자른 데이터셋 폴더에서 onset찾기
        do save onset_frames of dataset_train
        
        #2. apex를 추출해서 apex_output_dir에 저장.
        apex_frame = apex를 뽑아내는 함수
        do save apex_frames of dataset_train
        
        #3. AU를 추출해서 AU_output_dir에 저장.
        AU = extract_au
        AU(apex_output_dir, AU_output_dir)
        #do save AUs of dataset_train
        
        #4. label 정보 담긴 파일(.txt로) 생성
        # sample_name label 형식으로 2열로 만들기
        