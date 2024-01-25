from coder import image_to_dna,dna_to_image
from simulation import channel_simulation
from ssim import calculate_ssim

input_image = ["test_images\\10DPI_1.bmp"]
output_image = ["10DPI_1_output.bmp"]
encode_list = []
image_nums = len(input_image)
for i in range(image_nums):
    encode_list+=image_to_dna(input_image[i],i)



seq = channel_simulation(encode_list,5,False,{ "column": 0,"pi": 0.0,"pd": 0.0,"ps": 0.01 })

dna_to_image(seq,output_image,False)
