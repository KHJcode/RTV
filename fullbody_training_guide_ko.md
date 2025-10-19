# 전신 의류 학습 가이드

이 문서는 기존 상의(upper-body) 파이프라인을 확장하여 전신(full-body) 의류 학습을 수행하는 방법을 설명합니다. 데이터셋 준비부터 학습 실행, `demo.py`에서 체크포인트를 활용하는 절차까지 모두 다룹니다.

## 준비물
- 대상 의류를 착용한 상태로 촬영한 고해상도(가능하면 4K) 영상
- 각 프레임에 대한 의류 분할 마스크(알파 채널 포함 PNG 권장)
- `requirements.txt`에 명시된 의존성(이 저장소의 `rtv` conda 환경을 생성하면 데이터 생성과 학습에 모두 사용할 수 있습니다)

## 1단계 — 참고 영상 촬영
상의 학습 설명서(`training_instructions.md`)에서 제시한 포즈 가이드를 참고하여 여러 포즈를 포함하는 영상을 촬영합니다. 다양한 포즈를 캡처할수록 학습 성능이 좋아집니다.

## 2단계 — 마스크 생성
[muggled_sam](https://github.com/heyoeyo/muggled_sam)과 같은 도구를 사용하여 고품질의 의류 마스크를 생성합니다. 각 프레임 순서와 동일하게 정렬된 PNG(알파 채널 포함)로 내보냅니다.

## 3단계 — 전신용 데이터셋 생성
마스크 압축 파일을 원하는 폴더에 해제한 뒤 다음 명령을 실행합니다.

```bash
python DatasetGeneration/fullbody_dataset_generation.py \
  --video_path <path/to/video.mp4> \
  --mask_dir <path/to/mask_directory> \
  --dataset_name <garment_dataset_name>
```

이 스크립트는 `./PerGarmentDatasets/<garment_dataset_name>`에 전신 ROI(1024×768)로 정규화된 의류 이미지, DensePose 결과, SMPL 가상 마네킹, 전/역방향 아핀 변환을 저장합니다. `dataset_info.json`에는 원본 프레임 크기와 ROI 정보가 기록됩니다.

## 4단계 — 전신 체크포인트 학습
다음과 같이 학습을 실행합니다.

```bash
python Training/fullbody_training.py \
  --model pix2pixHD_RNN_RGBA \
  --input_nc 6 \
  --output_nc 4 \
  --batchSize 4 \
  --img_size 576 \
  --dataset_path ./PerGarmentDatasets/<garment_dataset_name> \
  --name <checkpoint_name> \
  --niter 80 \
  --niter_decay 80
```

- `img_size`는 ROI의 높이를 의미하며, 로더가 자동으로 너비를 계산해 4:3 비율(576×432)을 유지합니다.
- 여러 데이터셋을 동시에 학습하려면 `--dataset_path path_a,path_b`처럼 쉼표로 구분해 전달할 수 있습니다.
- 학습 결과와 체크포인트는 `./checkpoints/<checkpoint_name>` 폴더에 저장됩니다.

## 5단계 — 데모에서 테스트
`demo.py`의 `FullBodySeqFrameProcessor` 초기화 부분에서 체크포인트 이름을 새로 학습한 모델로 변경합니다.

```python
self.frame_processor = FullBodySeqFrameProcessor('<checkpoint_name>')
```

`rtv` 환경을 활성화한 뒤 다음 명령으로 실행합니다.

```bash
python demo.py --camera-rotate -90  # 카메라 방향에 따라 필요 시 값 조정
```

체크포인트가 정상적으로 로드되면 실시간 화면에서 새로운 전신 의류가 적용된 모습을 확인할 수 있습니다.

### 참고 사항
- 마스크와 영상 프레임 순서가 정확히 일치해야 학습 중 오류나 아티팩트가 발생하지 않습니다.
- 다른 해상도로 촬영한 경우에도 스크립트가 원본 크기를 자동으로 기록하고 변환을 처리합니다.
- 학습은 GPU 자원을 많이 사용하므로 VRAM 사용량을 모니터링하고 필요하면 `batchSize`를 줄여주세요.
