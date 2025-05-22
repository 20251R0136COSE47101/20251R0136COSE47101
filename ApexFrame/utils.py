import os
import cv2

def loadImages(path, dataset_type, onset=None, offset=None):
    """
    프레임 이미지가 저장된 폴더에서 순차적으로 이미지를 불러옴.
    CASMEII, SAMM: onset~offset 없이 전체 불러오기
    SMIC: onset~offset 지정 시 해당 범위만 잘라서 사용
    """
    files = sorted([f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')])
    
    # SMIC는 onset ~ offset 범위를 지정함
    if dataset_type == 'SMIC' and onset is not None and offset is not None:
        files = files[onset:offset+1]

    images = []
    for f in files:
        img = cv2.imread(os.path.join(path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # dlib은 RGB 이미지 사용
        images.append(img)

    return 0, images  # 0은 start index, 그냥 placeholder