import os
import numpy as np
import cv2

def extract_apex_onset_from_txt(txt_path, image_folder, output_folder):
    """
    txt_path: OF.txt 또는 OS.txt 파일 경로
    image_folder: 프레임 이미지가 저장된 폴더
    output_folder: onset/apex 이미지를 저장할 폴더
    """
    # (1) optical 값 불러오기
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    start_idx = int(lines[0].strip())  # 시작 인덱스 (보통 1)
    values = [float(line.strip().split()[0]) for line in lines[1:]]  # Global ROI만 사용

    values = np.array(values)
    
    # (2) apex frame: 최댓값 위치
    apex_relative_idx = np.argmax(values)
    apex_frame_idx = start_idx + apex_relative_idx

    # (3) onset frame: 처음으로 값이 threshold 넘는 지점 (혹은 첫 non-zero)
    threshold = np.max(values) * 0.15  # 15% 이상의 변화로 onset 판단
    onset_relative_idx = np.argmax(values > threshold)
    onset_frame_idx = start_idx + onset_relative_idx

    print(f"Onset Frame: {onset_frame_idx}, Apex Frame: {apex_frame_idx}")

    # (4) 이미지 저장
    for label, frame_idx in [('onset', onset_frame_idx), ('apex', apex_frame_idx)]:
        filename = f"{frame_idx:04d}.jpg"  # 프레임명 형식에 따라 조정
        src_path = os.path.join(image_folder, filename)
        dst_path = os.path.join(output_folder, f"{label}.jpg")

        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            cv2.imwrite(dst_path, img)
        else:
            print(f"Warning: Image file {filename} not found in {image_folder}")


def main(): #please fill in the path
    txt_path = "" #OS.txt route
    image_folder = "" #dataset route
    output_folder = ""

    extract_apex_onset_from_txt(txt_path, image_folder, output_folder)

if __name__ == "__main__":
    main()