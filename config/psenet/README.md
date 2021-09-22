# Shape Robust Text Detection with Progressive Scale Expansion Network
## Introduction
```
@inproceedings{wang2019shape,
  title={Shape Robust Text Detection with Progressive Scale Expansion Network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```


## Results and Models

- Total-Text

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | 87.3 | 77.9 | 82.3 | [config](psenet_r50_tt.py) | [model](https://drive.google.com/file/d/1Czu4Lc8vLSQ5FKm7d9G16e5PlyxPlxhq/view?usp=sharing) |
| PSENet (paper) | ResNet50 | N | 81.8 | 75.1 | 78.3 | - | - |
| PSENet | ResNet50 | Y | 89.3 | 79.6 | 84.2 | [config](psenet_r50_tt_finetune.py) | [model](https://drive.google.com/file/d/1h7P-BvD8f2FSn5t_jBEArSUxp5hFeYIb/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Y | 84.0 | 78.0 | 80.9 | - | - |

- CTW1500

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | 82.6 | 76.4 | 79.4 | [config](psenet_r50_ctw.py) | [model](https://drive.google.com/file/d/1J-YSnUHdTe2BBwU9e3Rx1z1je3jBtRfn/view?usp=sharing) |
| PSENet (paper) | ResNet50 | N | 80.6 | 75.6 | 78.0 | - | - |
| PSENet | ResNet50 | Y | 84.3 | 78.9 | 81.5 | [config](psenet_r50_ctw_finetune.py) | [model](https://drive.google.com/file/d/11Db47I75ZlF9203aIA6PBmbtPado90vU/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Y | 84.8 | 79.7 | 82.2 | - | - |

- ICDAR 2015

| Method | Backbone | Finetune | Scale | Precision (%) | Recall (%) | F-measure (%) | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | S: 736 | 83.6 | 74.0 | 78.5 | [config](psenet_r50_ic15_736.py) | [model](https://drive.google.com/file/d/1ZTdlCOmKmp-ZMCC5FdS89c5PsJMPoxkw/view?usp=sharing) |
| PSENet | ResNet50 | N | S: 1024 | 84.4 | 76.3 | 80.2 | [config](psenet_r50_ic15_1024.py) | [model](https://drive.google.com/file/d/11FCzOWlzE6toq2s6uuMR6XZeLmi5BE2-/view?usp=sharing) |
| PSENet (paper) | ResNet50 | N | L: 2240 | 81.5 | 79.7 | 80.6 | - | - |
| PSENet | ResNet50 | Y | S: 736 | 85.3 | 76.8 | 80.9 | [config](psenet_r50_ic15_736_finetune.py) | [model](https://drive.google.com/file/d/12YVKEMkIl_qcaGBVZRV8wTU81p6K9pix/view?usp=sharing) |
| PSENet | ResNet50 | Y | S: 1024 | 86.2 | 79.4 | 82.7 | [config](psenet_r50_ic15_1024_finetune.py) | [model](https://drive.google.com/file/d/1TENl7ng_m12SRm8TVQYfTTYqU-M33HiC/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Y | L: 2240 | 86.9 | 84.5 | 85.7 | - | - |

- SynthText

| Method | Backbone |            Config             |                           Download                           |
| :----: | :------: | :---------------------------: | :----------------------------------------------------------: |
| PSENet | ResNet50 | [config](psenet_r50_synth.py) | [model](https://drive.google.com/file/d/1blFDPLzV2NT4guYl-Jsm3zufcA96tRP1/view?usp=sharing) |

