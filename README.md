# HELIX
HELIX: A Novel Biological Image Storage System based on DNA Data Storage
## Requirements
- opencv-python
- reedsolo
- scikit-image
- numpy
- sewar
- torch
- torchvision
## Quick Start
You can execute the **test.py** file. It will encode an example image as a set of nucleotide sequences, simulate generating a 1% error, and successfully recover it.

## Data sets

You can download the images used in the HELIX test at https://drive.google.com/file/d/1-rivI0uMgyow0THctDZdn23O454hkma1/view?usp=sharing

These images are all from publicly available datasets.

Spatiotemporalomics dataset: https://www.science.org/doi/10.1126/science.abp9444

CT dataset:https://www.kaggle.com/datasets/kmader/siim-medical-images

X-ray dataset:https://www.kaggle.com/datasets/tommyngx/digital-knee-xray/data

Sequencing datasets for in vitro experiments:

## Parameter settings
If you want to use other images, please modify the parameters in **coder.py**:
- **WIDTH** Image width resolution
- **HEIGHT** Image high resolution
- **BLOCK_COUNT** Number of generated blocks, which is the product of length and width divided by 16 and rounded up

Other adjustable parameters include:
- **IMAGE_QUALITY** The clarity of the image after encoding
- **DATA_LENGTH** The length of the data bytes of each sequence
- **IMAGE_NUMS** The number of encoded images

## Image Repair 

If you want to enable the image repair function, please download the network from https://drive.google.com/file/d/1hwrLgfwms1bDXkYxM8D_hd3IU8xaqDsU/view?usp=sharing and put it in the  folder, then modify the **repair_mode** option in the **dna_to_image** function. If you want to train a new image repair network, please create a new folder called **train** and put the image set in it. Then run **train.py** in the nn folder.

## Golang Codecs
We have Golang's Error Correcting Code coder, which has a faster decoding speed. If you want to use Golang decoder, you need to overwrite all the files in the "go" folder to the root directory, and then change the **DATA_LENGTH** in **coder.py** to 10.

## Note
HELIX is still under review, we will update this document after acceptance and upload it to PyPI.
