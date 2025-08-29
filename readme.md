**Pix2day: Night-to-day Translation with Pix2Pix model**

This project is an implementation of EnlightenGAN for the task of image restoration, specifically focused on translating nighttime images into daytime images. The model learns to enhance low-light images, effectively converting them to a well-lit, daytime appearance.

**Project Overview**

The core of this project is the *EnlightenGAN.py* script, which defines the Generative Adversarial Network architecture. The goal is to train a generator model that can take a dark, nighttime image as input and produce a realistic, bright, daytime version of that image.

This repository provides the necessary scripts to prepare the dataset, train the model, and test its performance on new images.

**Directory Structure**
```bash
IMAGE-RESTORATION/
├── night2day/
│   ├── day/              # Target daytime images
│   └── night/            # Input nighttime images
├── .gitignore            # Specifies files to be ignored by Git
├── ds.py                 # Script for dataset handling and preprocessing
├── EnlightenGAN.py       # Core implementation of the EnlightenGAN model
├── readme.md             # Project documentation (this file)
├── test_model.py         # Script to test the trained generator
└── train.py              # Script to train the EnlightenGAN model
```

**Installation**

1. Clone the repository:

   ```bash
   git clone [https://github.com/your-username/IMAGE-RESTORATION.git](https://github.com/Omkaar-Kasodekar/IMAGE-RESTORATION.git)
   cd IMAGE-RESTORATION
   ```

2. Create a virtual environment (recommended):

   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install torch torchvision os PIL lpips piq
   ```

**Dataset** - [huggan/night2day](https://huggingface.co/datasets/huggan/night2day)

The dataset for this project is located in the night2day/ directory.

* The night/ folder should contain the low-light images that will be used as input to the model.
* The day/ folder should contain the corresponding ground truth, well-lit images.

The ds.py script is used to load and preprocess these images for training and testing. Ensure your image pairs are correctly placed in these folders before running the scripts.

