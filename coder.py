
import os
import pickle
import cv2
import numpy as np
import struct
import random
from PIL import Image
import torchvision.transforms as transforms
import ecc
from nn.Image_repair import Image_repair
from utils import UnNormalize
from RS import Outer_Code
import utils
WIDTH           = 4000
HEIGHT          = 8000
IMAGE_QUALITY   = 60
DUPLICATE       = 1
BLOCK_COUNT     = 125000
DATA_STRAND_GROUP_SIZE  = 32
DATA_LENGTH     = 11
IMAGE_NUMS = 1

OC = Outer_Code(fig_nums = IMAGE_NUMS)

def get_image_header(width, height, quality):
    seed_img = np.array([[[255, 255, 255]]])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality, int(cv2.IMWRITE_JPEG_RST_INTERVAL), 1]
    seed_img_encode = bytes(cv2.imencode('.jpg', seed_img, encode_param)[1])
    header_part1_index = seed_img_encode.find(bytes([0xFF,0xC0]))
    header_part2_index = seed_img_encode.find(bytes([0xFF,0xC4]))
    header_part3_index = seed_img_encode.find(bytes([0x00, 0x3F, 0x00]))
    header_part1 = seed_img_encode[:header_part1_index]
    header_part2 = seed_img_encode[header_part1_index:header_part2_index]
    header_part3 = seed_img_encode[header_part2_index:header_part3_index+3]
    header_part2 = header_part2[:5] \
                    + bytes([height//256, height%256, width//256, width%256]) \
                    + header_part2[9:]
    return header_part1 + header_part2 + header_part3

def image_encoding(filename, quality):
    img = cv2.imread(filename)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality, int(cv2.IMWRITE_JPEG_RST_INTERVAL), 1]
    result, encimg = cv2.imencode('.jpg', img, encode_param)

    jpeg_bin = bytes(encimg)
    sos_index = jpeg_bin.find(bytes.fromhex("FFDA"))

    data_start_index = sos_index + 14
    data_bytes = jpeg_bin[data_start_index:-2]
    return data_bytes

def get_image_blocks(jpeg_data):
    jpeg_blocks = []
    block_index = 0
    while True:
        next_index = jpeg_data.find(bytes([255,208+block_index%8]))
        if next_index == -1:
            jpeg_blocks.append(jpeg_data)
            break
        temp_block = jpeg_data[:next_index]
        jpeg_blocks.append(temp_block)
        # print(block_index, temp_block, len(temp_block))

        block_index += 1
        jpeg_data = jpeg_data[next_index+2:]
    return jpeg_blocks

def bytes_to_int(data):
    result = 0
    for i in data:
        result = result << 8
        result += i
    return result

def int_to_3bytes(x):
    if x < 0 or x > 16777216:
        raise Exception("Int out of range.")
    return bytes([x//65536, (x//256)%256, x%256])

def four_bits_xor(x):
    result = 0
    while x != 0:
        x, four_bits =divmod(x, 16)
        result = result ^ four_bits
    return result

def verify_index(index_with_check):
    index, check = divmod(index_with_check, 16)
    passed = (four_bits_xor(index) == check) and (index < BLOCK_COUNT)
    return passed, index

def index_add_check(index):
    temp_index = index << 4
    temp_index += four_bits_xor(index)
    return temp_index

def one_bit_xor(x):
    result = 0
    while x != 0:
        x, one_bit =divmod(x, 2)
        result = result ^ one_bit
    return result

def encode_strand_index(total, current):
    if total > 7 or current > 7:
        raise Exception("strand index exceed 7")
    result = 0
    result += total
    result = result << 1
    result += one_bit_xor(total)
    result = result << 3
    result += current
    result = result << 1
    result += one_bit_xor(current)
    return result

def verify_strand_index(strand_index):
    total_strand_index, current_strand_index = divmod(strand_index, 16)
    total, check = divmod(total_strand_index, 2)
    total_passed = (one_bit_xor(total) == check) and (total < 8)
    current, check = divmod(current_strand_index, 2)
    current_passed = (one_bit_xor(current) == check)  and (current < 8)
    total = total if total_passed else None
    current = current if current_passed else None
    return total_passed, total, current_passed, current

def two_bit_xor(x):
    result = 0
    while x != 0:
        x, one_bit =divmod(x, 4)
        result = result ^ one_bit
    return result

def encode_current_strand_index(current):
    if current > 63:
        raise Exception("current index exceed 63")
    return (current<<2) + two_bit_xor(current)

def verify_current_strand_index(encoded_current):
    current, check = divmod(encoded_current, 4)
    current_passed = (two_bit_xor(current) == check) and (current < 18)
    current = current if current_passed else None
    return current_passed, current

def encode_length(length):
    if length > 63:
        raise Exception("length exceed 63")
    return (length<<2) + two_bit_xor(length)

def verify_length(encoded_length):
    length, check = divmod(encoded_length, 4)
    length_passed = (two_bit_xor(length) == check) and (length < 18)
    length = length if length_passed else None
    return length_passed, length

def crc_checksum(data):
    polynomial = 0x07
    checksum = 0x00
    for byte in data:
        checksum ^= byte
        for _ in range(8):
            if checksum & 0x80:
                checksum = (checksum << 1) ^ polynomial
            else:
                checksum <<= 1
    return checksum & 0xFF

def verify_crc(data, check):
    temp_check = crc_checksum(data)
    return temp_check == check

def gen_crc4(data):
    data = struct.pack(">L", data)
    crc = 0x00
    polynomial = 0x13  # CRC-4 Polynomial: x^4 + x + 1
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
    return crc & 0x0F

def verify_crc4(data, check):
    temp_check = gen_crc4(data)
    return temp_check == check

def encode_enhanced_header(image_id, index, part, length):
    result = 0
    # image_id
    if  image_id < 0 or image_id > 4:
        raise ValueError(image_id)
    result += image_id
    # index
    if index < 0 or index > 131071:
        raise ValueError(index)
    result <<= 17

    result += index
    # part
    if part < 0 or part > 31:
        raise ValueError(part)
    result <<= 5
    result += part
    # length
    if length < 0 or length > 15:
        raise ValueError(length)
    result <<= 4
    result += length
    # Checksum
    result <<= 4
    result += gen_crc4(result)
    return struct.pack(">L", result)

def decode_enhanced_header(header):
    header_int = struct.unpack(">L", header)[0]
    data, check = divmod(header_int, 16)
    check_passed = verify_crc4(data<<4, check)
    if not check_passed:
        return None, None, None, None, None
    data, length = divmod(data, 1<<4)
    data, part = divmod(data, 1<<5)
    image_id, index = divmod(data, 1<<17)
    outer = False
    if index >= BLOCK_COUNT or part >= DATA_STRAND_GROUP_SIZE:
        outer = True
    return image_id, index, part, length, outer

def block_to_strand(start_index, data, global_runtime_part, image_id=0):
    strands = []
    while True:
        if len(data) >= (DATA_LENGTH*2):
            temp_data = encode_enhanced_header(image_id, start_index, global_runtime_part%DATA_STRAND_GROUP_SIZE, DATA_LENGTH) \
                        + data[:DATA_LENGTH] \
                        + bytes([crc_checksum(data[:DATA_LENGTH])])
            strands.append(temp_data)
            
            global_runtime_part += 1
            data = data[DATA_LENGTH:]

        elif DATA_LENGTH < len(data) < (DATA_LENGTH*2):
            cut_point = len(data) // 2

            temp_data = encode_enhanced_header(image_id, start_index, global_runtime_part%DATA_STRAND_GROUP_SIZE, cut_point) \
                        + data[:cut_point] \
                        + bytes([random.randint(0, 10) for _ in range((DATA_LENGTH-cut_point))]) \
                        + bytes([crc_checksum(data[:cut_point])])
            strands.append(temp_data)
            
            global_runtime_part += 1
            data = data[cut_point:]

        else:
            temp_data = encode_enhanced_header(image_id, start_index, global_runtime_part%DATA_STRAND_GROUP_SIZE, len(data)) \
                        + data \
                        + bytes([random.randint(0, 10) for _ in range((DATA_LENGTH-len(data)))]) \
                        + bytes([crc_checksum(data)])
            strands.append(temp_data)
            global_runtime_part += 1
            break
    # print([len(i) for i in strands])
    return strands

def blocks_to_strands(jpeg_blocks, image_id=0):
    runtime_strands = []
    hardcoded_strands = []
    start_index = 0
    end_index = 1
    global_runtime_part = 0
    while True:
        while end_index < len(jpeg_blocks) and jpeg_blocks[start_index] == jpeg_blocks[end_index]:
            end_index += 1
        temp_block = jpeg_blocks[start_index]

        temp_rt_strands = block_to_strand(start_index, temp_block, global_runtime_part, image_id)
        runtime_strands += temp_rt_strands
        global_runtime_part += len(temp_rt_strands)
            
        start_index = end_index
        end_index += 1
        if start_index == len(jpeg_blocks):
            break
    if len(hardcoded_strands) != 0:
        while len(hardcoded_strands[-1]) != 15:
            hardcoded_strands[-1] += bytes([158, 195, 0, 0, 0])
    convert_headers(runtime_strands,16)
    convert_headers(hardcoded_strands, 15)
    strands_data = runtime_strands + hardcoded_strands
    return strands_data

def convert_headers(strands, length):
    INDEX_X_Y = [27, 115, 118, 109, 119, 30, 98, 99, 78, 57, 34, 54, 56, 39, 50, 35, 45, 38, 44, 51, 55, 75, 121, 108, 114, 103, 120, 102, 135, 134, 136, 65, 146, 226, 149, 245, 137, 33, 224, 58, 144, 28, 100, 215, 63, 235, 208, 9, 126, 199, 127, 111, 210, 156, 10, 168, 6, 26, 67, 247, 254, 128, 176, 206, 74, 230, 221, 211, 157, 112, 185, 207, 141, 32, 253, 0, 246, 22, 196, 89, 11, 31, 152, 140, 200, 249, 5, 209, 25, 122, 1, 84, 197, 182, 36, 80, 212, 40, 72, 155, 183, 133, 192, 177, 41, 42, 60, 239, 23, 104, 24, 227, 202, 236, 52, 96, 240, 232, 163, 125, 173, 130, 53, 81, 243, 180, 4, 161, 150, 82, 83, 37, 8, 188, 186, 68, 116, 217, 110, 194, 167, 73, 222, 70, 234, 139, 166, 190, 233, 171, 179, 91, 201, 14, 29, 2, 160, 154, 71, 148, 93, 204, 94, 101, 69, 251, 158, 237, 59, 151, 123, 129, 198, 205, 175, 20, 105, 19, 216, 191, 86, 223, 76, 184, 61, 153, 164, 193, 49, 113, 16, 21, 132, 17, 169, 85, 117, 18, 47, 77, 219, 147, 15, 97, 181, 138, 213, 143, 203, 172, 252, 170, 62, 187, 12, 43, 46, 159, 107, 79, 250, 131, 87, 214, 165, 218, 66, 64, 92, 142, 225, 241, 88, 174, 231, 13, 90, 162, 106, 124, 189, 145, 242, 255, 244, 220, 48, 95, 229, 7, 228, 238, 178, 3, 248, 195]
    for i in range(len(strands)):
        strands[i] = bytes([INDEX_X_Y[x] for x in strands[i][:length]]) + strands[i][length:]
    # for i in range(len(strands)):
    #     strands[i] = bytes([INDEX_X_Y[x] for x in strands[i][:4]]) + strands[i][4:]

def revert_headers(strands, length):
    INDEX_Y_X = [75, 90, 155, 253, 126, 86, 56, 249, 132, 47, 54, 80, 214, 235, 153, 202, 190, 193, 197, 177, 175, 191, 77, 108, 110, 88, 57, 0, 41, 154, 5, 81, 73, 37, 10, 15, 94, 131, 17, 13, 97, 104, 105, 215, 18, 16, 216, 198, 246, 188, 14, 19, 114, 122, 11, 20, 12, 9, 39, 168, 106, 184, 212, 44, 227, 31, 226, 58, 135, 164, 143, 158, 98, 141, 64, 21, 182, 199, 8, 219, 95, 123, 129, 130, 91, 195, 180, 222, 232, 79, 236, 151, 228, 160, 162, 247, 115, 203, 6, 7, 42, 163, 27, 25, 109, 176, 238, 218, 23, 3, 138, 51, 69, 189, 24, 1, 136, 196, 2, 4, 26, 22, 89, 170, 239, 119, 48, 50, 61, 171, 121, 221, 192, 101, 29, 28, 30, 36, 205, 145, 83, 72, 229, 207, 40, 241, 32, 201, 159, 34, 128, 169, 82, 185, 157, 99, 53, 68, 166, 217, 156, 127, 237, 118, 186, 224, 146, 140, 55, 194, 211, 149, 209, 120, 233, 174, 62, 103, 252, 150, 125, 204, 93, 100, 183, 70, 134, 213, 133, 240, 147, 179, 102, 187, 139, 255, 78, 92, 172, 49, 84, 152, 112, 208, 161, 173, 63, 71, 46, 87, 52, 67, 96, 206, 223, 43, 178, 137, 225, 200, 245, 66, 142, 181, 38, 230, 33, 111, 250, 248, 65, 234, 117, 148, 144, 45, 113, 167, 251, 107, 116, 231, 242, 124, 244, 35, 76, 59, 254, 85, 220, 165, 210, 74, 60, 243]
    for i in range(len(strands)):
        strands[i] = bytes([INDEX_Y_X[y] for y in strands[i][:length]]) + strands[i][length:]
    # for i in range(len(strands)):
    #     strands[i] = bytes([INDEX_Y_X[y] for y in strands[i][:4]]) + strands[i][4:]

def revert_header(strand, length):
    INDEX_Y_X = [75, 90, 155, 253, 126, 86, 56, 249, 132, 47, 54, 80, 214, 235, 153, 202, 190, 193, 197, 177, 175, 191, 77, 108, 110, 88, 57, 0, 41, 154, 5, 81, 73, 37, 10, 15, 94, 131, 17, 13, 97, 104, 105, 215, 18, 16, 216, 198, 246, 188, 14, 19, 114, 122, 11, 20, 12, 9, 39, 168, 106, 184, 212, 44, 227, 31, 226, 58, 135, 164, 143, 158, 98, 141, 64, 21, 182, 199, 8, 219, 95, 123, 129, 130, 91, 195, 180, 222, 232, 79, 236, 151, 228, 160, 162, 247, 115, 203, 6, 7, 42, 163, 27, 25, 109, 176, 238, 218, 23, 3, 138, 51, 69, 189, 24, 1, 136, 196, 2, 4, 26, 22, 89, 170, 239, 119, 48, 50, 61, 171, 121, 221, 192, 101, 29, 28, 30, 36, 205, 145, 83, 72, 229, 207, 40, 241, 32, 201, 159, 34, 128, 169, 82, 185, 157, 99, 53, 68, 166, 217, 156, 127, 237, 118, 186, 224, 146, 140, 55, 194, 211, 149, 209, 120, 233, 174, 62, 103, 252, 150, 125, 204, 93, 100, 183, 70, 134, 213, 133, 240, 147, 179, 102, 187, 139, 255, 78, 92, 172, 49, 84, 152, 112, 208, 161, 173, 63, 71, 46, 87, 52, 67, 96, 206, 223, 43, 178, 137, 225, 200, 245, 66, 142, 181, 38, 230, 33, 111, 250, 248, 65, 234, 117, 148, 144, 45, 113, 167, 251, 107, 116, 231, 242, 124, 244, 35, 76, 59, 254, 85, 220, 165, 210, 74, 60, 243]
    return bytes([INDEX_Y_X[y] for y in strand[:length]]) + strand[length:]

def strands_to_blocks(strands):
    blocks = [None]*BLOCK_COUNT

    #for i in range(5):
    #    print(list(strands[i]))
    for strand in strands:

        # Run-time blocks

        ###########################################注意，我在这里把下面的防聚合物解码器给去掉了
        #strand = revert_header(strand, 15)
        image_index, start, current, length, outer = decode_enhanced_header(strand[:4])
        if image_index == None or outer == True:
            continue
        length = min(length, DATA_LENGTH)
        data = strand[4:4+length]
        if blocks[start] == None:
            blocks[start] = JpegBlock(start)
        blocks[start].add_strand(current, data)
        # Verify block data
        block_verified = verify_crc(data, strand[-1])
        if not block_verified:
            blocks[start].is_broken = True

    return blocks

def build_image_data(blocks):
    data = bytes()
    prev_valid_block = None
    broken_blocks = []
    for i in range(BLOCK_COUNT):
        temp_data = bytes()
        if blocks[i] != None:
            # print(i, blocks[i])
            temp_data = blocks[i].get_bytes()
            prev_valid_block = blocks[i]
        else:
            temp_data = prev_valid_block.get_bytes()
        
        if prev_valid_block.is_broken:
            broken_blocks.append(i)

        for j in range(len(temp_data)-1):
            if temp_data[j] == 0xff and temp_data[j+1] != 0x00:
                temp_data = temp_data[:j+1]+bytes([0x00])+temp_data[j+1:]

        data += temp_data
        if i != BLOCK_COUNT-1:
            data += bytes([255, 208+i%8])
    return data, broken_blocks

def build_image(data):
    JPEG_HEADER = get_image_header(WIDTH, HEIGHT, IMAGE_QUALITY)
    file_data = JPEG_HEADER + data + bytes([255, 217])
    return file_data

def build_save_bmp(jpeg_data, filename,i):
    image = cv2.imdecode(np.frombuffer(jpeg_data,dtype=np.dtype('uint8')), cv2.IMREAD_COLOR)
    cv2.imwrite(filename, image)

class JpegBlock():
    def __init__(self, index):
        self.index = index
        self.is_start = False
        self.is_end = False
        self.strands = [None]*DATA_STRAND_GROUP_SIZE
        self.is_broken = False

    def set_strand_length(self, length):
        if length > len(self.strands):
            self.strands += [bytes() for _ in range(length-len(self.strands))]

    def add_strand(self, current, data):
        self.strands[current] = data

    def get_bytes(self):
        searching_list = self.strands + self.strands
        start = 0
        end = 0
        max_length = 0
        max_sub_list = (0, 0)
        while start < len(searching_list) and end < len(searching_list):
            if start == end:
                if searching_list[start] == None:
                    start += 1
                    end += 1
                else:
                    end += 1
                continue
            if searching_list[end] != None:
                end += 1
            else:
                temp_max_length = end-start
                if temp_max_length > max_length:
                    max_length = temp_max_length
                    max_sub_list = (start, end)
                start = end
        result_sub_list = searching_list[max_sub_list[0]:max_sub_list[1]]
        result = bytes()
        for i in result_sub_list:
            result += i
        return result

    def __repr__(self):
        return str(self.is_start) + " " + str(self.is_end) + " " + str(self.strands)  + " " + str(self.dislocated)
    
def image_to_dna(input_image_path, image_id=0):
    print("Encoding Params:")
    print("IMAGE_QUALITY:", IMAGE_QUALITY, "DUPLICATE:", DUPLICATE)

    print("Compressing image...")
    raw_data = image_encoding(input_image_path, IMAGE_QUALITY)

    blocks = get_image_blocks(raw_data)
    print("Image block count:", len(blocks))

    print("Encoding image blocks...")
    orginal_strands_data = blocks_to_strands(blocks, image_id)
    print("Raw strand count:", len(orginal_strands_data))
    
    print("Encoding DNA strands...")
    out_sequences = OC.encode(image_id,orginal_strands_data)
    dna_sequences = ecc.encode(out_sequences, DUPLICATE)
    print("DNA strand count:", len(dna_sequences))
    return dna_sequences

def dna_to_image(dna_sequences, output_image_path,repair_mode = False):
    print("Decoding DNA strands...")
    strands = ecc.decode(dna_sequences, 2)
    #print("Decoded DNA strand count:", len(strands))
    test_f = open("test_f","w")
    test_f.write(str(strands))
    test_f.close()
    #test_g = open("test_f","r").read()
    #strands = eval(test_g)
    #print(len(strands))
    output_strands = OC.decode(IMAGE_NUMS,strands)
    print("Decoding image blocks...",len(output_strands[0]))
    for i in range(IMAGE_NUMS):
        dec_blocks = strands_to_blocks(output_strands[i])
                
        print("Rebuilding image...")
        dec_image_data, broken_blocks = build_image_data(dec_blocks)
        print("Broken blocks: ", broken_blocks)
        
        # Add AI repair code here #

        dec_file_data = build_image(dec_image_data)
        bmp_data = build_save_bmp(dec_file_data, output_image_path[i],i)
        if repair_mode == True :
            try:
                os.makedirs("repair/cropped")
                os.makedirs("repair/real")
                os.makedirs("repair/recon")
            except OSError:
                pass
            Repair_NN = Image_repair()
            img = Image.open(output_image_path[i])
            real_img = Image.open("1.bmp")
            new_img = img.copy()
            to_pil = transforms.ToPILImage()
            unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            for j,index in enumerate(broken_blocks):
                
                h = (index//250)*16
                w = (index%250)*16
                #x1, y1 = 100, 100
                #x2, y2 = 200, 200
                black_image = Image.new('RGB', [16,16], (0, 0, 0))
                
                now_img = img.crop((w-16,h-16,w+32,h+32))
                if j!= len(broken_blocks)-1:
                    if index == broken_blocks[j+1]-1:
                        img.paste(black_image, (w,h,w+16,h+16))
                        img.paste(black_image, (w+16,h,w+32,h+16))

                real_img_block = real_img.crop((w-16,h-16,w+32,h+32))
                real_img_block.save("repair/real/"+str(j)+"_"+str(index)+'.jpg')
                #now_img.show()
                repair_img = Repair_NN.repair_fuction(now_img,j,index)
                repair_img = unorm(repair_img)
                repair_img = to_pil(repair_img)
                new_img.paste(repair_img, (w-16,h-16,w+32,h+32))
            new_img.save("repair_"+output_image_path[i])        

    print("Done!")


