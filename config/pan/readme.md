# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network
## Introduction
```
@inproceedings{wang2019efficient,
  title={Efficient and accurate arbitrary-shaped text detection with pixel aggregation network},
  author={Wang, Wenhai and Xie, Enze and Song, Xiaoge and Zang, Yuhang and Wang, Wenjia and Lu, Tong and Yu, Gang and Shen, Chunhua},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8440--8449},
  year={2019}
}
```

Note that, the original PAN is based on Python 2.7 and Pytorch 0.4.1.
When migrating it to Python 3.6 and Pytorch 1.1.0, we make the following two changes to the default settings.
- Using Adam optimizer;
- PolyLR is also used in the pre-training phase.

## Results and Models
[Total-Text](https://github.com/cs-chan/Total-Text-Dataset)
| Method | Backbone | Fine-tuning | Precision (%) | Recall (%) | F-measure (%) | Model |
| - | - | - | - | - | - | - |
| PAN | ResNet18 | N | 87.9 | 79.6 | 83.5 | |
| PAN | ResNet18 | Y | 88.5 | 81.7 | 85.0 | |

[CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
| Method | Backbone | Fine-tuning | Precision (%) | Recall (%) | F-measure (%) | Model |
| - | - | - | - | - | - | - |
| PAN | ResNet18 | N | 85.1 | 79.1 | 82.0 | |
| PAN | ResNet18 | Y | 86.0 | 81.0 | 83.4 | |

[ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)
| Method | Backbone | Fine-tuning | Precision (%) | Recall (%) | F-measure (%) | Model |
| - | - | - | - | - | - | - |
| PAN | ResNet18 | N | 84.4 | 77.5 | 80.8 | |
| PAN | ResNet18 | Y | 86.6 | 79.7 | 83.0 | |

[MSRA-TD500](http://www.iapr-tc11.org/dataset/MSRA-TD500/MSRA-TD500.zip)
| Method | Backbone | Fine-tuning | Precision (%) | Recall (%) | F-measure (%) | Model |
| - | - | - | - | - | - | - |
| PAN | ResNet18 | N | 82.0 | 79.4 | 80.7 | |
| PAN | ResNet18 | Y | 85.7 | 83.4 | 84.5 | |
