## News
- (2021/04/08) PSENet and PAN are included in [MMOCR](https://github.com/open-mmlab/mmocr).

## Introduction
This repository contains the official implementations of [PSENet](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html), [PAN](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Efficient_and_Accurate_Arbitrary-Shaped_Text_Detection_With_Pixel_Aggregation_Network_ICCV_2019_paper.html) and [PAN++](https://arxiv.org/abs/2105.00405).

<details open>
<summary>Text Detection</summary>

- [x] [PSENet (CVPR'2019)](config/psenet/)
- [x] [PAN (ICCV'2019)](config/pan/)

</details>

<details open>
<summary>Text Spotting</summary>

- [x] [PAN++ (TPAMI'2021)](config/pan_pp)

</details>

## Installation

First, clone the repository locally:

```shell
git clone https://github.com/whai362/pan_pp.pytorch.git
```

Then, install PyTorch 1.1.0+, torchvision 0.3.0+, and other requirements:

```shell
conda install -c pytorch pytorch torchvision
pip install -r requirement.txt
```

Finally, compile codes of post-processing:

```shell
# build pse and pa algorithms
./compile.sh
```

## Dataset
Please refer to [dataset/README.md](dataset/README.md) for dataset preparation.

## Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ${CONFIG_FILE}
```
For example:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/pan/pan_r18_ic15.py
```

## Testing

### Evaluate the performance

```shell
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
cd eval/
./eval_{DATASET}.sh
```
For example:
```shell
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar
cd eval
./eval_ic15.sh
```

### Evaluate the speed

```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
For example:
```shell script
python test.py config/pan/pan_r18_ic15.py checkpoints/pan_r18_ic15/checkpoint.pth.tar --report_speed
```

## Citation

Please cite the related works in your publications if it helps your research:

##### PSENet

```
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```

##### PAN

```
@inproceedings{wang2019efficient,
  title={Efficient and accurate arbitrary-shaped text detection with pixel aggregation network},
  author={Wang, Wenhai and Xie, Enze and Song, Xiaoge and Zang, Yuhang and Wang, Wenjia and Lu, Tong and Yu, Gang and Shen, Chunhua},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8440--8449},
  year={2019}
}
```

##### PAN++

```
@article{wang2021pan++,
  title={PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Liu, Xuebo and Liang, Ding and Zhibo, Yang and Lu, Tong and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

## License

This project is developed and maintained by [IMAGINE Lab@National Key Laboratory for Novel Software Technology, Nanjing University](https://cs.nju.edu.cn/lutong/ImagineLab.html).

<img src="logo.jpg" alt="IMAGINE Lab">

This project is released under the [Apache 2.0 license](https://github.com/whai362/pan_pp.pytorch/blob/master/LICENSE).
