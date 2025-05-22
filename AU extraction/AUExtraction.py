# import os
# import libreface

# input_dir = "Test"      # apex frame folder route
# output_dir = "Test"      # .csv output directory
# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(input_dir):
#     if filename.lower().endswith((".jpg", ".png")):
#         input_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename.replace(".jpg", ".csv").replace(".png", ".csv"))
#         libreface.get_facial_attributes(
#             file_path=input_path,
#             output_save_path=output_path,
#             device="cuda:0"  # GPU 사용 (없으면 "cpu")
#         )
        
import os
import libreface

current_dir = os.getcwd()
input_dir = os.path.join(current_dir, "Test")     # apex frame folder route
output_dir = os.path.join(current_dir, "au_test")      # .csv output directory
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".jpg", ".csv").replace(".png", ".csv"))
        libreface.get_facial_attributes(
            file_path=input_path,
            output_save_path=output_path,
            device="cpu"  # GPU 사용 (없으면 "cpu")
        )