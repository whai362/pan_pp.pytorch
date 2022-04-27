# PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text
## Introduction
```
@article{wang2021pan++,
  title={PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Liu, Xuebo and Liang, Ding and Yang, Zhibo and Lu, Tong and Shen, Chunhua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```

## Results and Models

### Text Detection

- ICDAR 2015

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN++ detection only | ResNet18 | N  | 84.4 | 78.5 | 81.3 | [config](pan_pp_r18_ic15_736_det_only.py) | [model](https://drive.google.com/file/d/1XOxmiGiKfLsOGZ4z-O3uMv9HDkBuXjPH/view?usp=sharing) |

### End-to-End Text Spotting

- ICDAR 2015

| Method | Backbone | Finetune | Vocabulary | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN++ 736 joint train | ResNet18 | N | N | 83.6 | 54.0 | 65.6 | [config](pan_pp_r18_ic15_736_joint_train.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
| PAN++ 736 joint train | ResNet18 | N | G | 82.3 | 55.8 | 66.5 | [config](pan_pp_r18_ic15_736_joint_train_voc_g.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
| PAN++ 736 joint train | ResNet18 | N | W | 90.1 | 63.9 | 74.8 | [config](pan_pp_r18_ic15_736_joint_train_voc_w.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
| PAN++ 736 joint train | ResNet18 | N | S | 92.2 | 70.3 | 79.8 | [config](pan_pp_r18_ic15_736_joint_train_voc_s.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
_
| PAN++ 896 joint train | ResNet18 | N | N | 82.0 | 56.0 | 66.6 | [config](pan_pp_r18_ic15_736_joint_train.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
| PAN++ 896 joint train | ResNet18 | N | G | 81.5 | 57.6 | 67.5 | [config](pan_pp_r18_ic15_736_joint_train_voc_g.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
| PAN++ 896 joint train | ResNet18 | N | W | 90.1 | 63.9 | 74.8 | [config](pan_pp_r18_ic15_736_joint_train_voc_w.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |
| PAN++ 896 joint train | ResNet18 | N | S | 92.2 | 70.3 | 79.8 | [config](pan_pp_r18_ic15_736_joint_train_voc_s.py) | [model](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |


Todo:
- Models and configs on other datasets.

