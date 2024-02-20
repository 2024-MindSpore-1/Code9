# LOGO-MindSpore

## Requirements

- Python >= 3.6
- numpy
- scipy
- MindSpore >= 2.0.0
- torch_videovision
```
pip install git+https://github.com/hassony2/torch_videovision
```

## Datasets

- LOGO Dataset (Download link in README_LOGO.md)

### Data from Pretrained Model

- `video_feature_dict.pkl`: I3D backbone video feature
- `swin_features_dict.pkl`: SWIN backbone video feature
- `video_feamap_dict.pkl` (FineDiving-GOAT Only): video feature map
- `inception_feature_dict.pkl`: InceptionV3 backbone video frames feature
- `ob_result_new.pkl`: bounding boxes of actors in video frames、

[Download link](https://cloud.tsinghua.edu.cn/d/3a8267a34e514ba1bd7e/) (include model checkpoints)

## Training and Evaluation

### Training

```bash
bash ./scripts/train.sh <GPUIDS> --use_goat=1 --use_i3d_bb=0/1 --use_swin_bb=1/0 ...
```

### Evaluation

```bash
bash ./scripts/test.sh <GPUIDS> --use_goat=1 --use_i3d_bb=0/1 --use_swin_bb=1/0 --ckpts=<ckpt_path>
```

Same for all 3 models with scripts in their folders.

### Arguments

Parser path: 

- `CoRe-GOAT/utils/parser.py`
- `FineDiving-GOAT/utils.parser.py`
- `MUSDL-GOAT/MTL-AQA/config.py`

Arguments may need to modify:

- Data paths: data_root / label_path / boxes_path / i3d_feature_path / swin_feature_path / cnn_feature_path / train_split / test_split / formation_feature_path / feamap_root / ...
- Train config: lr/ bs_train / bs_test / workers / max_epoch / weight_decay / ...