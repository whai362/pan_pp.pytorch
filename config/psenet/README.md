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

| Method | Backbone | Finetune | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | [config](psenet_r50_tt.py) | 87.3 | 77.9 | 82.3 | [Google Drive](https://drive.google.com/file/d/1U8GK8BWdDOfz-p4Op4qqGJoEmnMQygpx/view?usp=sharing) |
| PSENet (paper) | ResNet50 | N | - | 81.8 | 75.1 | 78.3 | - | 
| PSENet | ResNet50 | Y | [config](psenet_r50_tt_finetune.py) | 89.3 | 79.6 | 84.2 | [Google Drive](https://drive.google.com/file/d/1CSwtB6T70VFyz_xQBDxM-1ym70OIao-k/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Y | - | 84.0 | 78.0 | 80.9 | - | 

- CTW1500

| Method | Backbone | Finetune | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | [config](psenet_r50_ctw.py) | 82.6 | 76.4 | 79.4 | [Google Drive](https://drive.google.com/file/d/1AeUj_E6tKzo4uAvwNLQ98Tf2bmASxdv0/view?usp=sharing) |
| PSENet (paper) | ResNet50 | N | - | 80.6 | 75.6 | 78 | - | 
| PSENet | ResNet50 | Y | [config](psenet_r50_ctw_finetune.py) | 84.5 | 79.2 | 81.8 | [Google Drive](https://drive.google.com/file/d/1c0h6rzBB_T8eR_gt3xuvguEVJz2FVfNf/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Y | - | 84.8 | 79.7 | 82.2 | - | 

- ICDAR 2015

| Method | Backbone | Finetune | Scale | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | Shorter Side: 736 | [config](psenet_r50_ic15_736.py) | 83.6 | 74.0 | 78.5 | [Google Drive](https://drive.google.com/file/d/1kxnoYyLnMr_uhvso2v27We6gYNKANXER/view?usp=sharing) |
| PSENet | ResNet50 | N | Shorter Side: 1024 | [config](psenet_r50_ic15_1024.py) | 84.4 | 76.3 | 80.2 | [Google Drive](https://drive.google.com/file/d/1Yz4zrSpvt5nVIqT75EafBPwEl19Sj3Vg/view?usp=sharing) |
| PSENet (paper) | ResNet50 | N | Longer Side: 2240 | - | 81.5 | 79.7 | 80.6 | - | 
| PSENet | ResNet50 | Y | Shorter Side: 736 | [config](psenet_r50_ic15_736_finetune.py) | 85.3 | 76.8 | 80.9 | [Google Drive](https://drive.google.com/file/d/1flNt1L4cxXTzKc75NpPjfdBotNYOcQL6/view?usp=sharing) |
| PSENet | ResNet50 | Y | Shorter Side: 1024 | [config](psenet_r50_ic15_1024_finetune.py) | 86.2 | 79.4 | 82.7 | [Google Drive](https://drive.google.com/file/d/1nR0j7WBiyrpa1OF7GXzrbR2mrKP-PdiX/view?usp=sharing) |
| PSENet (paper) | ResNet50 | Y | Longer Side: 2240 | - | 86.9 | 84.5 | 85.7 | - | 



