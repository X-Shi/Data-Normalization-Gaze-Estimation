## Data Normalization for ETH-XGaze

The codes follow the paper [**'Revisiting Data Normalization for Appearance-Based Gaze Estimation'**](https://perceptualui.org/publications/zhang18_etra.pdf).

To normalize the training data, please run this line:\
`python nor_eye_face.py`

To normalize the testing data, please run this line:\
`python nor_eye_face_test.py`

Please obtain the [**ETH-XGaze dataset**](https://files.ait.ethz.ch/projects/xgaze/xucongzhang2020eccv.pdf) through this [link](https://ait.ethz.ch/xgaze). We do not provide the access to the dataset.

Please change the path to your raw dataset and the target path for the normalized dataset in file through *data_dir* and *output_dir*.
To obatin normalized image of resolution 448x448, please change the *patch_size* parameter. 

By changing *subject_begin* and *subject_end* through *-sb* and *-se* command parameters, it is possible to specify the subjects to be normalized.