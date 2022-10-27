# paddle_mobileone
PaddlePaddle implementation of [MobileOne](https://arxiv.org/abs/2206.04040). This repo is built on the basis of [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) framework.

## Benchmark

Comming soon.

## Training
- Single card training
```
python train.py -c  configs/MobileOne/MobileOne.yaml
```

- Distributed multi-card training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.7 -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 tools/train.py -c configs/MobileOne/MobileOne.yaml
```
## Evaluation

```
python eval.py -c  configs/MobileOne/MobileOne.yaml
```
