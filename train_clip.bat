@echo off
REM 设置环境变量
set PATH=%PATH%;D:\CODE\Anaconda3_ML\Anaconda\envs\yolov10\Scripts

REM 执行 Python 训练脚本
python -m open_clip_train.main ^
    --save-frequency 1 ^
    --zeroshot-frequency 1 ^
    --report-to tensorboard ^
    --train-data "file://D:/CODE/PyTorch/PyTorch_project1/chuanxinshiijan_test/CLIP_train_test/wds_dataset/train.tar" ^
    --val-data "file://D:/CODE/PyTorch/PyTorch_project1/chuanxinshiijan_test/CLIP_train_test/wds_dataset/val.tar" ^
    --dataset-type "webdataset" ^
    --train-num-samples 399 ^
    --val-num-samples 80 ^
    --warmup 10000 ^
    --batch-size 64 ^
    --lr 1e-3 ^
    --wd 0.1 ^
    --epochs 30 ^
    --workers 0 ^
    --model "ViT-B-32"

pause
