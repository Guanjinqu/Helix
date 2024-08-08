from coder import image_to_dna,dna_to_image
from simulation import channel_simulation
from tqdm import tqdm
import shutil


input_image = ["test_images/1.bmp"]
output_image = ["1_output.bmp"]
encode_list = []

image_nums = len(input_image)
for i in range(image_nums):
    encode_list+=image_to_dna(input_image[i],i)


try:
    shutil.rmtree("repair/")
except:
    pass

#encoding
for k in range(image_nums):
    encode_list+=image_to_dna(input_image[k],k)
print(len(encode_list[0]))
#simulated channel
seq = channel_simulation(encode_list,1,False,{ "column": 0,"pi": 0,"pd": 0,"ps": 0.01 })

#decoding
dna_to_image(seq,output_image,False)

