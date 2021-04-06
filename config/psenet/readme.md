# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network
## Introduction
```
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```


## Results and Models
[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)

| Method | Backbone | Scale | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | Shorter Side: 736 | [psenet_r50_ic15_736.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ic15_736.py) | 83.6 | 74.0 | 78.5 | [Google Drive](https://drive.google.com/file/d/1kxnoYyLnMr_uhvso2v27We6gYNKANXER/view?usp=sharing) |
| PSENet | ResNet50 | Shorter Side: 1024 | [psenet_r50_ic15_1024.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ic15_1024.py) | 84.4 | 76.3 | 80.2 | [Google Drive](https://drive.google.com/file/d/1Yz4zrSpvt5nVIqT75EafBPwEl19Sj3Vg/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Longer Side: 2240 | - | 81.5 | 79.7 | 80.6 | - | 

[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)

| Method | Backbone | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | [psenet_r50_ctw.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_ctw.py) | 82.6 | 76.4 | 79.4 | [Google Drive](https://drive.google.com/file/d/1AeUj_E6tKzo4uAvwNLQ98Tf2bmASxdv0/view?usp=sharing) |
| PSENet (paper) | ResNet50 | - | 80.6 | 75.6 | 78 | - | 

[Total-Text](https://github.com/cs-chan/Total-Text-Dataset)

| Method | Backbone | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | [psenet_r50_tt.py](https://github.com/whai362/PSENet/blob/python3/config/psenet/psenet_r50_tt.py) | 87.3 | 77.9 | 82.3 | [Google Drive](https://drive.google.com/file/d/1U8GK8BWdDOfz-p4Op4qqGJoEmnMQygpx/view?usp=sharing) |
| PSENet (paper) | ResNet50 | - | 81.8 | 75.1 | 78.3 | - | 