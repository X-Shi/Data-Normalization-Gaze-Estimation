## Data Normalization for ETH-XGaze

The codes follow the paper [**'Revisiting Data Normalization for Appearance-Based Gaze Estimation'**](https://perceptualui.org/publications/zhang18_etra.pdf).

To obtain the normalized dataset, Please change the data path to your downloaded raw dataset (*/eth-xgaze/raw) and the target path in file through *data_dir* and *output_dir*.
The code normalizes raw images into 224x224 resolution by default. To obatin normalized images of resolution 448x448, please change the *patch_size* parameter. 

To normalize the training data, please run this line:\
`python nor_eye_face.py`

To normalize the testing data, please run this line:\
`python nor_eye_face_test.py`

Please obtain the [**ETH-XGaze dataset**](https://files.ait.ethz.ch/projects/xgaze/xucongzhang2020eccv.pdf) through this [link](https://ait.ethz.ch/xgaze). We cannot provide the access to the dataset.

By changing *subject_begin* and *subject_end* through *-sb* and *-se* command parameters, it is possible to specify the subjects to be normalized.