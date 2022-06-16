# B993288_musics
 


오류 수난기

처음에 pytorch로 시작했지만
파일이 전혀 읽히지 않았습니다
wav파일을 npz로 바꿨을때 어떻게 처리되는지에 대한 지식이 없었기에
최대한 tensorflow에서 코드변경을 덜하는 선에서 해보자는 생각으로
tensorflow를 공부하면서 하게 되었는데
하다보니 그 코드가 tensorflow 1 버전이었고
현재 버전에서는 공간 할당 등 많은 변수가 바뀌어
사용할 수 없더군요
(다운그레이드를 사용해봤지만 다른쪽에서 빨간줄이 쳐지는 바람에,
이 코드는 어떻게든 쓸 수 없다는 결론을 내렸습니다)
결국 tensorflow 2.0과 keras까지 추가적으로 공부해서
지금의 코드가 나왔습니다.
하다보니 엉뚱하게도 전처리 파일 형식이 잘못되어 있음을 발견했습니다.
pythorch로 그냥 진행했으면 될 것이었는데 돌아오기엔 너무 먼 산이었어요.

100번이 넘어가는 디버깅 속에서,
거의 대부분은 전처리한 파일을 읽어들이지 못하는 현상,
그리고 제가 모르는 메서드의 반환값과 필요값들을 찾는데 보냈습니다.

0. 이거 뭐야.. 전처리 왜 안돼.. 
fshape를 shape로 고치니 돌아는 간다.
그런데... 예제파일 변환에 2시간이 걸림
몇번 실행하다가 아무것도 안나와서 당황했지만
print문을 중간중간 넣고
터질 때마다 다시 작업하는 식으로 진행

1. 안된다.. 예제코드가!
--> tensorflow 1,2의 코드가 혼합되어있음
--> 텐서플로우를 공부해서 파이토치로 변환해보자
--> tensorflow 1의 공간할당이 난해해서 실패
--> tensorflow 2로 코드를 변환한 후, keras를 이용해
최대한 단순하게 재현해보자
from tensorflow.keras import datasets, layers, models
안됨 : 이유는 pip을 써서 그럼
conda를 써서 tensorflow와 keras를 install
안됨 : conda를 업그레이드해야 함
설치 중 오류: 저장공간 부족이었음.
설치후 안됨: pip과 conda에 있는 패키지가 둘다 불러와져서 충돌
pip에 있던 것을 uninstall, 
conda tensorflow를 다운그레이드함

--> 레이어 input값을 넣는것에서 난관
#ic.summary() 안됨
Input을 넣거나, compile되기 전에는 summary를 할 수 없음

conv2d relu (3,3)
conv2d relu (3,3)
conv2d relu (3,24)

--> 어찌저찌 한 후 테스트용 파일값이 안맞음
--> 고치던 도중 메모리가 부족한지
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.12 GiB for an array with shape (299452608,) and data type float32
라는 오류를 띄움.
당시 메모리 100% 꽉참
리부팅함

ValueError: Input 0 of layer conv2d is incompatible with the layer: : expected min_ndim=4, found ndim=3. Full shape received: (256, 168, 174)
https://stackoverflow.com/questions/63279168/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expect

shape = None은 불가능하다.
ValueError: Please provide to Input a `shape` or a `tensor` or a `type_spec` argument. Note that `shape` does not include the batch dimension.

reshape가 원하는 모양대로 안나옴 --> mnist 예제 돌려가면서 비슷하게 만듬
Conv2D가 자꾸 4차원 입력값으로 넣으라고 함 --> 알파채널값이 안들어가서 reshape함
accuracy가 낮은 epoch에서 너무 낮음 --> 아마 분류할 값이 너무 많아서 초기에 부족한 거인듯?

test acc, loss가 그래프로 안나옴 --> 해결 못했음

matplotlib 안됨
-> import os 한 후에 충돌하지 않도록 os.environ['KMP_DUPLICATE_LIB_OK']='True' 구문 넣어줌
