# PubLayNet

PubLayNet is a large dataset of document images, of which the layout is annotated with both bounding boxes and polygonal segmentations. For more information, see [PubLayNet original](https://github.com/ibm-aur-nlp/PubLayNet)


<img src="./example_images/PMC4334925_00006.jpg" width=400> | <img src="./example_images/PMC538274_00004.jpg" width=400> 
:-------------------------:|:-------------------------:
**PMC4334925_00006.jpg**  | **PMC538274_00004.jpg**



## Recent updates 

`29/Feb/2020` - Add benchmarking for maskrcnn_resnet50_fpn.

`22/Feb/2020` - Pre-trained Mask-RCNN model in (Pytorch) are [released](maskrcnn) .



## Benchmarking

| Architecture  | Iter_num (x16) | AP | AP50 | AP75 | AP Small | AP Medium | AP Large | MD5SUM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [MaskRCNN-Resnet50-FPN](https://drive.google.com/file/d/1Jx2m_2I1d9PYzFRQ4gl82xQa-G7Vsnsl/view?usp=sharing)  | 196k  | 0.91| 0.98 | 0.96 | 0.41 | 0.76 | 0.95 | 393e6700095a673065fcecf5e8f264f7 |


## Inference


Download trained weights here, locate it in [maskrcnn directory](maskrcnn)


- [12000x16 iterations](https://drive.google.com/open?id=1T2ciEJ7npW_aBpNrKHiUAluyk04K0AWK)
- [50000x16 iterations](https://drive.google.com/open?id=1vl3XAYbGKlv70SNPReStZQ6I0Z9v1CSW)
- [120000x16 iterations](https://drive.google.com/open?id=13fhd_SS7fLrjLrCjVpCwOYGt_SlQ_7FW)
- [161000x16 iterations](https://drive.google.com/open?id=1KNOyw_D980bvFKb8U8NPPt-NWSsWJDe6)
- [174000x16 iterations](https://drive.google.com/open?id=13fhd_SS7fLrjLrCjVpCwOYGt_SlQ_7FW)
- [200000x16 iterations](https://drive.google.com/open?id=1rJ3fowtxGIcORzIZbQe9ibHN0ORoqkLN)


Run
```
python infer.py <path_to_image>
```

## Avarage Precision in validation stages (via Tensorboard)

<img src="https://user-images.githubusercontent.com/24642166/75600546-066b6900-5ae3-11ea-9774-a0a0396e6fb1.png" width=1000>


