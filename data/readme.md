data:

├─apex_output

├─AU_output

├─onset_output

├─dataset

│ 

└─label


위 구조대로 apex, au, onset이 저장되도록 하기.
dataset/train, dataset/test를 통해 각 frame을 찾아내어 저장.
각 사진들이 순서대로 들어와야하니, 저장시에 순서가 바뀌지 않게, 이름 같게 저장되게 해주세요.
(ex. 같은 영상에서 뽑은 apex, au, onset은 모두 0001.jpg처럼 같은 이름으로 각 폴더에 저장)
dataloader에서도 shuffle=True하는건 배치단위로 섞이니까 괜찮습니다.

+AU관련 확인할 것. 
AU결과가 csv로 나오는데 쓸데없는거 많잖아. 그거 전처리에서 필요한 것들만 남기고 다 지우는게 편할듯
각 column이 각 AU가 되도록 해주면 될 것 같습니다.
ex) 
AU_result.csv
AU1 | AU2 | AU3 | ~~
 1  |  0  |  1  | ~~
