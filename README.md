# CMDepth: Self-Supervised Monocular Depth Estimation based on Hybrid CNN-State Space Models

## Pretrained Models

Download pretrained models from the links below:

| Model | Resolution | Link |
|-------|------------|------|
| CMDepthres50 | 320x1024 | [Google Drive](https://drive.google.com/drive/folders/1Ad1CJhXWKHWg4MMhFAmVQCngVT_j-tOb?usp=drive_link) |

To evaluate a model on KITTI, run:

```bash
python evaluate_depth_config.py args_files/cm_args/kitti/CMDepthres50_320x1024.txt
python evaluate_depth_config.py args_files/cm_args/kitti/CMDepthres50_192x640.txt
```

## Inference with your own iamges KITTI
```bash
python test_simple_SQL_config.py ./conf/cvnXt.txt

## Training

Training code is currently being cleaned up and will be released together with additional pretrained models and detailed instructions on how to train.




