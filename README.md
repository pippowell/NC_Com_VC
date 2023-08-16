# Applying an SNN to Visual Data Classification

This repository contains the code for the masters thesis project Do You See What I See? - Applying a Spiking Neural Network to Visual Data Classification.

This project applied NeuCube (an SNN) and EEGChannelNet and an LSTM model to the visual classification EEG dataset created for and used by the latter two in the publications discussing and testing these models (references below). 

Where code from other authors was used, this is noted. 

Code marked as coming from the comparison models code is based off code from the following repository, which also contains a link to the dataset files: 
[Comparison Models Repository](https://github.com/perceivelab/eeg_visual_classification)

Code marked as coming from the NeuCube code is based off code from the NeuCube team's Python implementation of this network, which has not yet been made publicly available. References for the NeuCube network are also included below. 

The Data Preparation Files directory contains code used for reviewing, reformatting, and quartering the dataset as described in the thesis paper also included in this repository. 

The Master Training Files directory contains the master training files for NeuCube and for the comparison models (EEGChannelNet and 3 versions of the LSTM).

An additional final report file is also included, which outputs an aggregated summary of the networks' performance in their individual training rounds. 

The architecture files for these models are not included here, however the files for the comparison models may be found in the linked repository above.

Data files and result files for this thesis may be found in its OSF entry at this [link](https://osf.io/9e6r3/?view_only=6cc6c37e43ff47c38d87c3230178c29c).

</br>


**References**

_EEGChannelNet_

Palazzo, S., Spampinato, C., Kavasidis, I., Giordano, D., Schmidt, J., & Shah, M. (2021). Decoding
Brain Representations by Multimodal Learning of Neural Activity and Visual Features. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 43(11), 3833–3849. https://doi.org/10.1109/TPAMI.2020.2995909

_LSTM_

Spampinato, C., Palazzo, S., Kavasidis, I., Giordano, D., Souly, N., & Shah, M. (2017). Deep Learning
Human Mind for Automated Visual Classification. 2017 IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), 4503–4511. https://doi.org/https://doi.org/10.1109/CVPR.2017.479

_NeuCube_

Kasabov, N. (2012). Neucube evospike architecture for spatio-temporal modelling and pattern recognition of brain signals. Artificial Neural Networks in Pattern Recognition: 5th INNS IAPR TC 3
GIRPR Workshop, ANNPR 2012, Trento, Italy, September 17-19, 2012. Proceedings 5, 225–243.
https://doi.org/https://doi.org/10.1007/978-3-642-33212-8_21

Kasabov, N. K. (2014). Neucube: A spiking neural network architecture for mapping, learning and
understanding of spatio-temporal brain data. Neural Networks, 52, 62–76. https://doi.org/10.1016/j.neunet.2014.01.006

