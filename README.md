# Data Normalization - Gaze Estimation
This is a repository for the code used to generate input data for the paper *Investigation of Architectures and Receptive Fields for Appearance-based Gaze Estimation*. The main code repository can be found at [https://github.com/yunhanwang1105/GazeTech](https://github.com/yunhanwang1105/GazeTech)

This repository contains code for normalizing ETH-XGaze, MPIIFaceGaze, and Gaze360 datasets with single face in resolutions of 224x224 and 448x448, and multi-region crops of face and two eyes for resolution 224x224.
The separate code for different datasets can be found in individual folders. 

We refer to the paper ['Revisiting Data Normalization for Appearance-Based Gaze Estimation'](https://www.perceptualui.org/publications/zhang18_etra.pdf) for data normalization in the gaze estimation task. An updated code for that paper can be found [here](https://github.com/xucong-zhang/data-preprocessing-gaze). We also re-used the part of the code from [ETH-XGze repository](https://github.com/xucong-zhang/ETH-XGaze).

If you use the results, pre-trained models, or method from our arXiv paper, please cite it as: 
'TODO' our arXiv paper citation.

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
