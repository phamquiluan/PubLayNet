# PubLayNet

PubLayNet is a large dataset of document images, of which the layout is annotated with both bounding boxes and polygonal segmentations. For more information, see [PubLayNet original](https://github.com/ibm-aur-nlp/PubLayNet)


<img src="./example_images/PMC4334925_00006.jpg" width=400> | <img src="./example_images/PMC538274_00004.jpg" width=400> 
:-------------------------:|:-------------------------:
**PMC4334925_00006.jpg**  | **PMC538274_00004.jpg**



## Recent updates 

`15/Sept/2020` - Add training code.

`29/Feb/2020` - Add benchmarking for `maskrcnn_resnet50_fpn`.

`22/Feb/2020` - Pre-trained Mask-RCNN model in (Pytorch) are [released](maskrcnn) .



## Benchmarking

| Architecture  | Iter num (x16) | AP | AP50 | AP75 | AP Small | AP Medium | AP Large | MD5SUM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [MaskRCNN-Resnet50-FPN](https://drive.google.com/file/d/1Jx2m_2I1d9PYzFRQ4gl82xQa-G7Vsnsl/view?usp=sharing)  | 196k  | 0.91| 0.98 | 0.96 | 0.41 | 0.76 | 0.95 | 393e6700095a673065fcecf5e8f264f7 |


## Demo

Download trained weights in Benchmarking section above, locate it in [maskrcnn directory](maskrcnn)

Run
```
cd maskrcnn
python infer.py <path_to_image>
```

## Avarage Precision in validation stages (via Tensorboard)

<img src="https://user-images.githubusercontent.com/24642166/75600546-066b6900-5ae3-11ea-9774-a0a0396e6fb1.png" width=1000>


## Training

Please take a look at `training_code` dir. Sorry for the dirty code but I really don't have time to refactor it :D 
