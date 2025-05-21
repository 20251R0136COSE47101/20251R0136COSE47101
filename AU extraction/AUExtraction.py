import os
import libreface

input_dir = "C:/Users/Juna/20251R0136COSE47101/Test"      # apex frame folder route
output_dir = "C:/Users/Juna/20251R0136COSE47101/Test"      # .csv output directory
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jpg", ".csv").replace(".png", ".csv"))
        libreface.get_facial_attributes(
            file_path=input_path,
            output_save_path=output_path,
            device="cuda:0"  # GPU 사용 (없으면 "cpu")
        )