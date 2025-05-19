import cv2
import numpy as np
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import mediapipe as mp

# Mediapipe 얼굴 특징점 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def detect_landmarks(img):
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    return np.array([(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark])

def calculate_motion_score(frame1, frame2):
    # 얼굴 특징점 추출
    lmk1 = detect_landmarks(frame1)
    lmk2 = detect_landmarks(frame2)
    
    if lmk1 is None or lmk2 is None:
        return 0.0
    
    # 특징점 이동 거리 계산 (정규화)
    motion = np.linalg.norm(lmk1 - lmk2, axis=1).mean()
    return motion

def find_apex_onset(frames_dir):
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    # print(len(frame_files))
    motion_scores = []
    
    # 첫 프레임을 기준 프레임으로 설정
    base_frame = cv2.imread(frame_files[0])
    prev_frame = base_frame
    
    for frame_file in tqdm(frame_files[1:]):
        curr_frame = cv2.imread(frame_file)
        # Motion 계산 (Optical Flow + 특징점 이동)
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag_flow = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
        
        # 특징점 기반 motion 점수
        landmark_score = calculate_motion_score(prev_frame, curr_frame)
        
        # 종합 motion 점수
        combined_score = mag_flow * 0.7 + landmark_score * 0.3
        motion_scores.append(combined_score)
        
        prev_frame = curr_frame
    
    # Apex 프레임 탐지 (최대 motion 지점)
    apex_idx = np.argmax(motion_scores) + 1  # 기준 프레임 다음부터
    
    # Onset 프레임 탐지 (motion 임계값 초과 지점)
    threshold = np.mean(motion_scores) * 1.5
    onset_idx = next(i for i, score in enumerate(motion_scores) if score > threshold)
    
    return {
        'onset_frame': frame_files[onset_idx],
        'apex_frame': frame_files[apex_idx],
        'motion_curve': motion_scores
    }

result = find_apex_onset('dataset/Test/Test/Lie/Atul/What is your Name')
print(f"Apex Frame: {result['apex_frame']}")
print(f"Onset Frame: {result['onset_frame']}")
