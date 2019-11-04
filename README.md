# PGU-net-Model PyTorch1.0 Python3.7 with multi GPUs

This is my contribution to the "PGU-net+: Progressive Growing of U-net+ for Automated Cervical Nuclei Segmentation" in MICCAI2019 workshop MMMI.

Paper link
 Please cite this paper if you find this project helpful for your research.

# Dependencies

python 3.7, CUDA 10.1, numpy 1.17, matplotlib 3.1.1, scikit-image (0.21), scipy (0.3.1), pyparsing (2.4.2), pytorch (1.0) (anaconda is recommended)
other packages could be the latest version.

# Instructions for runing

# Training:

# 1.Install all dependencies

# 2.For PGU-net+: Progressive Growing of U-net+,
python train_PG.py
For the U-net structure in the ﬁnal stag,
python train.py

# Testing:
For testing,
python test.py
