## Prepare datasets
It is recommended to symlink the dataset root to $PAN_PP.PYTORCH/data. If your folder structure is different, you may need to change the corresponding paths in dataloader files.
```none
pan_pp.pytorch
└── data
    ├── CTW1500
    │   ├── train
    │   │   ├── text_image
    │   │   └── text_label_curve
    │   └── test
    │       ├── text_image
    │       └── text_label_curve
    ├── total_text
    │   ├── Images
    │   │   ├── Train
    │   │   └── Test
    │   └── Groundtruth
    │       ├── Polygon
    │       └── Rectangular
    ├── ICDAR2015
    │   └── Challenge4
    │       ├── ch4_training_images
    │       ├── ch4_training_localization_transcription_gt
    │       ├── ch4_test_images
    │       └── ch4_test_localization_transcription_gt
    ├── MSRA-TD500
    │   ├── train
    │   └── test
    ├── HUST-TR400
    ├── COCO-Text
    │   └── train2014
    ├── SynthText
    │   ├── 1
    │   ├── ...
    │   └── 200
    └── ICDAR2017-MLT
        ├── ch8_training_images
        ├── ch8_validation_images
        ├── ch8_training_localization_transcription_gt_v2
        └── ch8_validation_localization_transcription_gt_v2
```

## Download

These datasets can be downloaded from the following links:

- CTW1500 [[dataset]](https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk)
- Total-Text [[image]](https://drive.google.com/file/d/1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2/view?usp=sharing) [[gt]](https://drive.google.com/file/d/19quCaJGePvTc3yPZ7MAGNijjKfy77-ke/view?usp=sharing)

- ICDAR2015 [[dataset]](https://rrc.cvc.uab.es/?ch=4&com=downloads)

- MSRA-TD500 [[dataset]](http://www.iapr-tc11.org/dataset/MSRA-TD500/MSRA-TD500.zip)
- HUST-TR400 [[dataset]](http://mc.eistar.net/UpLoadFiles/dataset/HUST-TR400.zip)

- SynthText [[dataset]](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

- ICDAR2017-MLT [[dataset]](https://rrc.cvc.uab.es/?ch=8&com=downloads)

- COCO-Text [[dataset]](https://rrc.cvc.uab.es/?ch=5&com=downloads)

