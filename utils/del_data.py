import os

# CSV 파일들이 있는 폴더 경로
csv_dir = 'data/AU_output_rev'  # 예: 'C:/data/csv'
# JPG 파일들이 있는 폴더 경로
jpg_dir = 'data/onset_output'  # 예: 'C:/data/images'

# 파일 이름 번호 범위 (0001 ~ 0190)
start = 1
end = 190

for i in range(start, end + 1):
    num_str = f"{i:04}"  # 4자리 문자열, 예: 0001
    csv_path = os.path.join(csv_dir, f"{num_str}.csv")
    jpg_path = os.path.join(jpg_dir, f"{num_str}.jpg")

    if not os.path.exists(csv_path):
        if os.path.exists(jpg_path):
            os.remove(jpg_path)
            print(f"삭제됨: {jpg_path}")
        else:
            print(f"(이미 없음): {jpg_path}")

print("✅ 누락된 CSV에 대응되는 JPG 삭제 완료.")
