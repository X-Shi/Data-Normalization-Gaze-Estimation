## Data Normalization for MPIIFaceGaze

The codes follow the paper [**'Revisiting Data Normalization for Appearance-Based Gaze Estimation'**](https://perceptualui.org/publications/zhang18_etra.pdf).

To normalize the MPIIFaceGaze dataset, plead put the file *normalize_data.py* in the MPIIFaceGaze folder. You also need the files from MPIIGaze dataset. They are the files in the folder of 'Evaluation Subset/sample list for eye image' and the '.mat' file. 

To use the codes, please run:\
`python normalize_data.py`

Please obtain the datasets through these links: [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) and [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild).

## Attention
**The coordinate system of the gaze labels of the [original MPIIFaceGaze dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation) is different from our data normalization repository. The coordinate system in our data normalization repository is aligned with the one in [ETH-XGaze dataset](https://ait.ethz.ch/xgaze).
The difference between the two coordinate systems is that the pitch and yaw directions are opposite from the corresponding ones.**
