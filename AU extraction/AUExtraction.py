import os
import libreface

input_dir = ""      # apex frame folder route
output_dir = ""      # .csv output directory
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jpg", ".csv").replace(".png", ".csv"))
        libreface.get_facial_attributes(
            image_or_video_path=input_path,
            output_save_path=output_path,
            device="cuda:0"  # GPU 사용 (없으면 "cpu")
        )