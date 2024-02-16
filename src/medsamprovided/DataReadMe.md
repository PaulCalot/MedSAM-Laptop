# Dataset for the CVPR 2024 Challenge: MedSAM on Laptop

## Training Data

The training data contained 11 modalities, including Computed Tomography (CT), Magnetic Resonance Imaging (MRI), Positron Emission Tomography (PET), X-ray, ultrasound, mammography, Optical Coherence Tomography (OCT), endoscopy, fundus, dermoscopy, and microscopy.


All the images are from public datasets with a License for redistribution. To reduce dataset size, all the labeled slices are extracted and preprocessed into `npz` files. Each `npz` file contains
- `imgs`: image data; shape: (H,W,3) for 2D images and (D,H,W) for 3D CT, MR, and PET; Intensity range: [0, 255]
- `gts`: ground truth; shape: (H, W) for 2D images and (D,H,W) for 3D CT, MR, and PET;
- `spacing`: only for 3D CT, MR, PET

> Please refer to [MedSAM paper](https://www.nature.com/articles/s41467-024-44824-z#Sec8) for the preprocessing details


(Optional) We also provide a script to convert the `npz` files to `npy` files, which can be used for training the (Lite)MedSAM model
- step 1. Run `python npz2npy.py`
- step 2. Follow the [guideline](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/README.md#model-training) to train the baseline (LiteMedSAM) 



## Highly recommended using [public datasets](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?usp=sharing) 

We tried to release as many preprocessed datasets as we could but the majority of the public datasets don’t allow us to re-distribute them due to the license limitation. However, we highly recommend using the publicly accessible datasets and pre-trained models during model developments. We will maintain a list of public datasets and pre-trained models that can be used for this challenge. To make sure all participants can access these datasets and models, please fill out the following link to register the public datasets and models that you plan to use (Deadline: April 15, 2024).


- Register public datasets: https://forms.gle/MNdkQ273KXmF9PQv6

- Register public models: https://forms.gle/R11ZweUEbRtx9DQS7

- List of suggested puglic datasets and models: [link](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?usp=sharing)


We also provide scripts for pre-processing

- [3D images](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/pre_CT_MR.py)
- [2D images](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/pre_grey_rgb.py)


## Testing Set

We are working with multiple medical centers to curate a large-scale and diverse testing dataset, which covers all the above mentioned modalities as well as various anatomies and pathologies. The testing set format is also `npz`. 


- [Here](https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq?usp=sharing) are some examples for testing data. 


- Please check this [demo](https://github.com/bowang-lab/MedSAM/blob/LiteMedSAM/README.md#quick-tutorial-on-making-submissions-to-cvpr-2024-medsam-on-laptop-challenge) for inference and evaluation: 

