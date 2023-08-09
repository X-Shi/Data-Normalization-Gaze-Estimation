## Data Normalization for Gaze360

The codes follow the paper [**'Revisiting Data Normalization for Appearance-Based Gaze Estimation'**](https://perceptualui.org/publications/zhang18_etra.pdf).

To normalize the data, please use this line:\
`python data_processing_gaze360.py --subject_index 0`\
Please replace the subject index with number from 0 to 79 (except 68).

You also need some files from ETH-XGaze dataset, which are 'cam00.xml' and 'face_model.txt'.

Please obtain the [**Gaze360 dataset**](http://gaze360.csail.mit.edu/iccv2019_gaze360.pdf) through this [link](http://gaze360.csail.mit.edu/). We do not provide the access to the dataset.

Please change the path to your dataset and the target path for the normalized dataset in file *data_processing_gaze360.py*.
