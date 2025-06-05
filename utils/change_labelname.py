import os
import shutil

onset_path = 'C:/Users/edwar/Desktop/onset_output'  # 사진 폴더 경로로 변경
apex_path = 'C:/Users/edwar/Desktop/apex_output'
res_onset_path = 'data\\onset_output_rev'
res_apex_path = 'data\\apex_output_rev'

def rename_files(folder_path, res_path):
    # 파일 목록을 정렬하여 충돌 방지 (역순으로 변경)
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')], reverse=True)
    print(len(file_list))
    for filename in file_list:
        num = int(filename.split('.')[0])
        new_num = num - 1
        new_filename = f'{new_num:04d}.jpg'
        shutil.copy2(
            os.path.join(folder_path, filename),
            os.path.join(res_path, new_filename)
        )   
    print('파일 이름 변경 완료')

if __name__ == "__main__":
    rename_files(onset_path, res_onset_path)
    rename_files(apex_path, res_apex_path)