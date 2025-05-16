ApexFrame.py
- 프레임 단위로 분할된 이미지 데이터셋에서 Optical Strain 값을 txt 파일로 추출
- CUSTOM dataset을 사용할 경우 파일 가장 밑 main에서 running() 함수 안의 값들을 수정(DataSet type = 'CUSTOM')

ExtractImage.py
- ApexFrame.py에서 추출괸 Optical Strain 파일(OS.txt)을 가지고 실제 apexframe과 onset frame에 해당하는 이미지를 추출
- main() 파일 안 path들을 채워서 실행할 것