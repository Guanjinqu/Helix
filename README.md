# HELIX
HELIX: A Novel Biological Image Storage System based on DNA Data Storage
## Requirements
- opencv-python
- reedsolo
- numpy
- torch
- torchvision
## Quick Start
You can execute the **test.py** file. It will encode an example image as a set of nucleotide sequences, simulate generating a 1% error, and successfully recover it.
You can download the images used in the HELIX test at https://drive.google.com/file/d/10TlTANyCy9x_YeySMbfHEgeZ9nunW6yZ/view?usp=sharing. These images are all from publicly available datasets.

If you want to use other images, please modify the parameters in **coder.py**:
- **WIDTH** Image width resolution
- **HEIGHT** Image high resolution
- **BLOCK_COUNT** Number of generated blocks, which is the product of length and width divided by 16 and rounded up

Other adjustable parameters include:
- **IMAGE_QUALITY** The clarity of the image after encoding
- **DATA_LENGTH** The length of the data bytes of each sequence
- **IMAGE_NUMS** The number of encoded images

If you want to enable the image repair function, please download the network from https://drive.google.com/file/d/1hwrLgfwms1bDXkYxM8D_hd3IU8xaqDsU/view?usp=sharing and put it in the **nn** folder, then modify the **repair_mode** option in the **dna_to_image** function. If you want to train a new image repair network, please create a new folder called **train** and put the image set in it. Then run **train.py** in the nn folder.

## Note
HELIX is still under review, we will update this document after acceptance and upload it to PyPI.
