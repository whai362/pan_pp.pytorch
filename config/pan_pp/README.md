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

| Method | Backbone | Finetune | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN++ | ResNet18 | N  | [config](pan_pp_r18_ic15.py) | 84.4 | 78.5 | 81.3 | [Google Drive](https://drive.google.com/file/d/1XOxmiGiKfLsOGZ4z-O3uMv9HDkBuXjPH/view?usp=sharing) |

### End-to-End Text Spotting

- ICDAR 2015

| Method | Backbone | Finetune | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PAN++ | ResNet18 | N | [config](pan_pp_r18_ic15_joint_train.py) | 83.6 | 54.1 | 65.6 | [Google Drive](https://drive.google.com/file/d/1HQ6LKVyuS5xcvU9IfdMJSC5ogCzCL4K6/view?usp=sharing) |

Todo:
- Models and configs on other datasets.

