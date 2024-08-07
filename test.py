from coder import image_to_dna,dna_to_image
from simulation import channel_simulation
from tqdm import tqdm
import os
import cv2
import shutil
input_image = ["test_images/1.bmp"]
output_image = ["1_output.bmp"]
encode_list = []
image_nums = len(input_image)
for i in range(image_nums):
    encode_list+=image_to_dna(input_image[i],i)
ssim_list = []
repair_list=[]
error_list = [0.02,0.02,0.02]


try:
    shutil.rmtree("repair/")
except:
    pass



for k in range(image_nums):
    encode_list+=image_to_dna(input_image[k],k)
    #print(len(encode_list))
seq = channel_simulation(encode_list,1,False,{ "column": 0,"pi": 0,"pd": 0,"ps": 0.000 })

dna_to_image(seq,output_image,True)

