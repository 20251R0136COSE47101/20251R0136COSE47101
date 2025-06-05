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

au_info = {
    "au_1": "R", "au_2": "R", "au_4": "R", "au_5": "R", "au_6": "R", "au_7": "D",
    "au_9": "R", "au_10": "D", "au_12": "R", "au_14": "D", "au_15": "R", "au_17": "R",
    "au_20": "R", "au_23": "D", "au_24": "D", "au_25": "R", "au_26": "R"
}

au_presence_cols = [au for au, pred in au_info.items()]
au_intensity_cols = [f"{au}_intensity" for au, pred in au_info.items() if pred == "R"]
expected_columns = au_presence_cols + au_intensity_cols

def fill_in_missing_au(au_dir):
    for filename in os.listdir(au_dir):
        file_path = os.path.join(au_dir, filename)
        try:
            df = pd.read_csv(file_path)

            # 누락된 AU 열은 0으로 채움
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0

            # 정해진 순서대로 정렬
            df = df[expected_columns]

            # 덮어쓰기 저장
            df.to_csv(file_path, index=False)
            #print(f"✅ 정리 완료: {filename}")

        except Exception as e:
            print(f"error")



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
            print(output_path)
            libreface.get_facial_attributes(
                file_path=input_path,
                output_save_path=output_path,
                weights_download_dir="AU/weights_libreface",
                device="cpu"  # GPU 사용 (없으면 "cpu")
            )

    filter_csv(output_dir)
    fill_in_missing_au(output_dir)
