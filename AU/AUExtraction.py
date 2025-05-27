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
import pandas as pd



def filter_csv(output_dir): #AU 관련 정보만 남기기
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            df = pd.read_csv(file_path)

            au_columns = [col for col in df.columns if col.startswith("au_")]
            df_au = df[au_columns]

            df_au.to_csv(file_path, index=False)
            #print(f"AU 정보만 필터됨: {filename}")

        except Exception as e:
            print(f"error")



def extract_au(apex_dir, au_dir):
    #current_dir = os.getcwd()
    #input_dir = os.path.join(current_dir, "Test")     # apex frame folder route
    #output_dir = os.path.join(current_dir, "au_test")     # .csv output directory
    #os.makedirs(output_dir, exist_ok=True)
    input_dir = apex_dir
    output_dir = au_dir    

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".png")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".jpg", ".csv").replace(".png", ".csv"))
            libreface.get_facial_attributes(
                file_path=input_path,
                output_save_path=output_path,
                device="cpu"  # GPU 사용 (없으면 "cpu")
            )

    filter_csv(output_dir)

