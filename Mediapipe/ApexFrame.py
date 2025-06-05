import cv2
import numpy as np
import os
from tqdm import tqdm
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
# preventing conflict with kaggle dataset preprocess
idx = 69 

def detect_landmarks(img):
    if img is None:
        print("Err")
        return None
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    h, w = img.shape[:2]
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark])

def calculate_motion_score(prev_frame, curr_frame):
    # Optical Flow
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag_flow = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()

    # Landmark Displacement
    lmk1 = detect_landmarks(prev_frame)
    lmk2 = detect_landmarks(curr_frame)
    if lmk1 is None or lmk2 is None:
        return mag_flow  # Fallback to optical flow only
    landmark_motion = np.linalg.norm(lmk1 - lmk2, axis=1).mean()

    # Combined Score
    combined_score = mag_flow * 0.7 + landmark_motion * 0.3
    return combined_score

"""
def find_onset_apex_frames(frames_dir):
    frame_files = sorted([os.path.join(frames_dir, f)
                          for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if len(frame_files) < 2:
        print("Not enough frames to analyze.")
        return None

    motion_scores = []
    prev_frame = cv2.imread(frame_files[0])
    for frame_file in tqdm(frame_files[1:], desc="Processing frames"):
        curr_frame = cv2.imread(frame_file)
        if curr_frame is None or prev_frame is None:
            motion_scores.append(0)
            prev_frame = curr_frame
            continue
        score = calculate_motion_score(prev_frame, curr_frame)
        motion_scores.append(score)
        prev_frame = curr_frame

    motion_scores = np.array(motion_scores)
    print(motion_scores)
    apex_idx = np.argmax(motion_scores) + 1  # +1 due to offset in motion_scores
    threshold = np.mean(motion_scores) * 1.5
    onset_idx = next((i for i, score in enumerate(motion_scores) if score > threshold), 0)

    return {
        'onset_frame': frame_files[onset_idx],
        'apex_frame': frame_files[apex_idx],
        'motion_scores': motion_scores
    }

if __name__ == "__main__":
    dataPath = '../dataset/Test/Test/Lie/Atul/What is your Name'
    testPath = dataPath + '/Test/Test'
    labelList = os.listdir(testPath)
    for label in labelList: # Lie, Truth
        personPath = testPath + '/' + label
        personList = os.listdir(personPath)
        for p in personList:
            subjectPath = personPath + '/' + p
            subjectList = os.listdir(subjectPath)
            if not os.path.isdir(subjectPath):
                continue

            MEList = os.listdir(subjectPath)
            for ME in MEList: # traverse all Micro-Expression sequences of the subject
                print('Calculating OF/OS in ' + subject + '/' + ME)
                MEPath = subjectPath + '/' + ME
                '''
                MEResPath = subResPath + '/' + ME
                if os.path.exists(MEResPath): continue
                if not os.path.exists(MEResPath):
                    os.makedirs(MEResPath)
                '''
                startIdx, imgs = loadImages(MEPath, type)
                
                OS, OF = extractFeature(imgs, 0.2, type)
                # drawPic(OF, MEResPath, 'Flow', startIdx, 0)
                # drawPic(OS, MEResPath, 'Strain', startIdx, 0)
                saveData('OF.txt', OF, startIdx)
                saveData('OS.txt', OS, startIdx)
    result = find_onset_apex_frames(frames_directory)
    if result:
        print(f"Onset Frame: {result['onset_frame']}")
        print(f"Apex Frame: {result['apex_frame']}")
"""

# frames_dir must contain set of frames
def find_onset_apex_frames(frames_dir, onset_output_dir, apex_output_dir):
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if len(frame_files) < 2:
        print("Not enough frames to analyze.")
        return None

    # print(len(frame_files))
    motion_scores = []
    prev_frame = cv2.imread(frame_files[0])
    for frame_file in tqdm(frame_files[1:], desc="Processing frames"):
        curr_frame = cv2.imread(frame_file)
        if curr_frame is None or prev_frame is None:
            motion_scores.append(0)
            prev_frame = curr_frame
            continue
        score = calculate_motion_score(prev_frame, curr_frame)
        motion_scores.append(score)
        prev_frame = curr_frame

    motion_scores = np.array(motion_scores)
    print(motion_scores)
    apex_idx = np.argmax(motion_scores) + 1  # +1 due to offset in motion_scores
    threshold = np.mean(motion_scores) * 1.5
    onset_idx = next((i for i, score in enumerate(motion_scores) if score > threshold), 0)

    """
    return {
        'onset_frame': frame_files[onset_idx],
        'apex_frame': frame_files[apex_idx],
        'motion_scores': motion_scores
    }
    """
    global idx
    idx += 1
    filename = f"{idx:04d}.jpg"
    print(filename)
    idx += 1
    #frame
    onset_frame = frame_files[onset_idx]
    apex_frame = frame_files[apex_idx]
    #path
    onset_path = os.path.join(onset_output_dir, filename)
    apex_path = os.path.join(apex_output_dir, filename)

    cv2.imwrite(onset_path, cv2.imread(onset_frame))
    cv2.imwrite(apex_path, cv2.imread(apex_frame))


