# Data Normalization - Gaze Estimation
This is a repository for the code used to generate input data for the paper *Investigation of Architectures and Receptive Fields for Appearance-based Gaze Estimation*. The main code repository can be found at [https://github.com/yunhanwang1105/GazeTech](https://github.com/yunhanwang1105/GazeTech)

This repository contains code for normalizing ETH-XGaze, MPIIFaceGaze, and Gaze360 datasets with single face in resolutions of 224x224 and 448x448, and multi-region crops of face and two eyes for resolution 224x224.
The separate code for different datasets can be found in individual folders. 

We refer to the paper ['Revisiting Data Normalization for Appearance-Based Gaze Estimation'](https://www.perceptualui.org/publications/zhang18_etra.pdf) for data normalization in the gaze estimation task. An updated code for that paper can be found [here](https://github.com/xucong-zhang/data-preprocessing-gaze). We also re-used the part of the code from [ETH-XGze repository](https://github.com/xucong-zhang/ETH-XGaze) and [FAZE repository](https://github.com/NVlabs/few_shot_gaze).


## Attention
**The coordinate system of the gaze labels of the [original MPIIFaceGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) is different from our data normalization repository. The coordinate system in our data normalization repository is aligned with the one in [ETH-XGaze dataset](https://ait.ethz.ch/xgaze).
The difference between the two coordinate systems is that the pitch and yaw directions are opposite from the corresponding ones.**

## Citation
If you use the results, pre-trained models, or method from our arXiv paper, please cite it as: 
```
@misc{wang2023investigation,
      title={Investigation of Architectures and Receptive Fields for Appearance-based Gaze Estimation}, 
      author={Yunhan Wang and Xiangwei Shi and Shalini De Mello and Hyung Jin Chang and Xucong Zhang},
      year={2023},
      eprint={2308.09593},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

If you only use the data normalization part, please cite the data normalization paper:
```
@inproceedings{zhang2018revisiting,
  title={Revisiting data normalization for appearance-based gaze estimation},
  author={Zhang, Xucong and Sugano, Yusuke and Bulling, Andreas},
  booktitle={Proceedings of the 2018 ACM symposium on eye tracking research \& applications},
  pages={1--9},
  year={2018}
}
```
