    _   _                      _   _   _      _                      _
    | \ | |                    | | | \ | |    | |                    | |
    |  \| | ___ _   _ _ __ __ _| | |  \| | ___| |___      _____  _ __| | __
    | . ` |/ _ \ | | | '__/ _` | | | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
    | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   <
    \_| \_/\___|\__,_|_|  \__,_|_| \_| \_/\___|\__| \_/\_/ \___/|_|  |_|\_\

^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V
SECTION 1: INTRODUCTION
^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V

This is the README document for the CO395 - Neural Network Coursework. Our group
consists of the following members.
- Emil Soerensen
- Finn Bauer
- Tristan Clark
- Yuhling Lee


^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V
SECTION 2: INSTALLATION
^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V

1) Ensure your device has Python 3.6 installed
2) Install the required dependencies for Python
  - Note: we opted to use the PyTorch library for implementation
3) Navigate to the folder containing nn_lib.py


^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V
SECTION 3: USAGE
^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V^V

When you have followed the steps above you should now be able to:
1) Use the neural network mini library                             - nn_lib.py
2) Run the pre trained forward model on default dataset            - learn_FM.py
3) Run the pre trained region of interest model on default dataset - learn_ROI.py

To customize the models:

1. PATH SETTING
To load a custom dataset we've set up an easy to configure path string you just
need to update at the top of the learn_FM.py and learn_ROI.py files:

  # Load Dataset
  dataset_path = "./FM_dataset.dat"

2. CHANGE MODEL
By default, the learn_FM.py and learn_ROI.py files are set up to use the best
performing found during this study. To change the model simply change the model
path the is set up at the top of the files:

  # Load Model
  model_path = "./model_FM.pth"

3. RUN MODEL
Run both models from the command line to see performance metrics.
  - Note: predict_hidden function is available in both models.
