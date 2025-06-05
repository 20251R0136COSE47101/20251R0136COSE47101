import os
import re
import pandas as pd

# CSV 파일이 들어있는 폴더 경로
csv_folder = 'data/AU_output_rev'

# 출력할 라벨 파일
output_file = 'labels.txt'

# 결과를 저장할 리스트
results = []

# 폴더 내 모든 4자리 숫자 .csv 파일 처리
for filename in os.listdir(csv_folder):
    if filename.endswith('.csv') and filename[:4].isdigit():
        num_str = filename[:4]
        num = int(num_str)

        # 라벨 조건
        if num < 35:
            label = 1
        elif num < 69:
            label = 0
        elif num < 130:
            label = 1
        elif num < 190:
            label = 0
        else:
            continue  # 190 이상은 제외

        results.append(f"{num_str} {label}")

# label.txt로 저장
with open(output_file, 'w') as f:
    for line in results:
        f.write(line + '\n')

print(f"✅ 완료: {len(results)}개 항목이 '{output_file}'에 저장되었습니다.")