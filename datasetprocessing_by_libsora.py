import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import itertools

spt = []
ins = []
n = 0
for instrument, note in itertools.product(range(128), range(50)):
    # 128*50번, 총 6400번 루프를 돕니다. 이사실을 모르고 프린트문을 안넣었다가 고장난 줄 알았어요.
    y, sr = librosa.load('GeneralMidi.wav', sr=None, offset=n, duration=2.0)
    print("libsora.load successfully load \n")
    # y값은 np.ndarray형태의 값이 나오게 됩니다. 오디오 시계열이며 다중 채널이 지원됩니다.
    # libsora.load는 (path *, sr,mono. offset, duration,dtype=<class 'numpy.float32'>, res_type = 'kaiser_best')
    # 의 형태를 띠고 있습니다.
    # offset은 소리의 시작 지점을 뜻합니다. n을 정했으므로 n의 값에 따라 읽는 지점이 달라질 것입니다.
    # sr(sampling rate, 목표 샘플링 속도를 뜻하며 스칼라값을 지닙니다.)값은 None으로 지정했으므로, 기본값인 22050이 됩니다.
    # duration = 2.0 n초 지점부터 2초까지만 데이터를 읽어옵니다.
    n += 2
    # 데이터를 늘리기 위해 white 노이즈를 섞은 버전도 함께 변환합니다
    # 시간 대역 데이터를 옥타브당 24단계로, 총 7옥타브로 변환할 겁니다.
    print("out loops, spt: ", spt)
    print("out loops, ins: ", ins)
    for r in (0, 1e-4, 1e-3):
        print("r processing, r =  ", r,"instrument = ", instrument, "note = ", note)
        # r은 0,0.001,0.0001의 값을 한번씩 가집니다. 6400*3. 우와. 19200번이나 루프를 돌아요.
        ret = librosa.cqt(y + ((np.random.rand(*y.shape) - 0.5) * r if r else 0), sr,
                          hop_length=1024, n_bins=24 * 7, bins_per_octave=24)
        # 주파수의 위상은 관심없고, 세기만 보겠으니 절대값을 취해줍니다
        ret = np.abs(ret) # np.abs() 함수는 제곱근에 제곱을 취하는 방식으로 절댓값을 취합니다.
        spt.append(ret)  # 스펙토그램을 저장합니다
        ins.append((instrument, 38 + note))  # 악기 번호와 음 높이를 저장합니다.
        print("inner loops, spt: ", spt)
        print("inner loops, ins: ", ins)

for note in range(46):
    y, sr = librosa.load('GeneralMidi.wav', sr=None, offset=n, duration=2.0)
    n += 2
    print("r2 processing, n = ",n)

    for r, s in itertools.product([0, 1e-5, 1e-4, 1e-3], range(7)):
        print("r processing, r = ", r, ", s = ", s)
        ret = librosa.cqt(y + ((np.random.rand(*y.shape) - 0.5) * r * s if r else 0), sr,
                          hop_length=1024, n_bins=24 * 7, bins_per_octave=24)
        ret = np.abs(ret)
        spt.append(ret)
        ins.append((note + 128, 0))

    # 아래의 코드는 변환된 주파수 대역의 스펙토그램을 보여줍니다.
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(ret), ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()

spt = np.array(spt, np.float32)
ins = np.array(ins, np.int16)
#np.savez('cqt2.npz', spec=spt, instr=ins)