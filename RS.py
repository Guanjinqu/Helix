#RS error correction code
import math

from reedsolo import RSCodec
import numpy as np
import struct

def revert_header(strand, length):
    INDEX_Y_X = [75, 90, 155, 253, 126, 86, 56, 249, 132, 47, 54, 80, 214, 235, 153, 202, 190, 193, 197, 177, 175, 191, 77, 108, 110, 88, 57, 0, 41, 154, 5, 81, 73, 37, 10, 15, 94, 131, 17, 13, 97, 104, 105, 215, 18, 16, 216, 198, 246, 188, 14, 19, 114, 122, 11, 20, 12, 9, 39, 168, 106, 184, 212, 44, 227, 31, 226, 58, 135, 164, 143, 158, 98, 141, 64, 21, 182, 199, 8, 219, 95, 123, 129, 130, 91, 195, 180, 222, 232, 79, 236, 151, 228, 160, 162, 247, 115, 203, 6, 7, 42, 163, 27, 25, 109, 176, 238, 218, 23, 3, 138, 51, 69, 189, 24, 1, 136, 196, 2, 4, 26, 22, 89, 170, 239, 119, 48, 50, 61, 171, 121, 221, 192, 101, 29, 28, 30, 36, 205, 145, 83, 72, 229, 207, 40, 241, 32, 201, 159, 34, 128, 169, 82, 185, 157, 99, 53, 68, 166, 217, 156, 127, 237, 118, 186, 224, 146, 140, 55, 194, 211, 149, 209, 120, 233, 174, 62, 103, 252, 150, 125, 204, 93, 100, 183, 70, 134, 213, 133, 240, 147, 179, 102, 187, 139, 255, 78, 92, 172, 49, 84, 152, 112, 208, 161, 173, 63, 71, 46, 87, 52, 67, 96, 206, 223, 43, 178, 137, 225, 200, 245, 66, 142, 181, 38, 230, 33, 111, 250, 248, 65, 234, 117, 148, 144, 45, 113, 167, 251, 107, 116, 231, 242, 124, 244, 35, 76, 59, 254, 85, 220, 165, 210, 74, 60, 243]
    return bytes([INDEX_Y_X[y] for y in strand[:length]]) + strand[length:]
def convert_headers(strands, length):
    INDEX_X_Y = [27, 115, 118, 109, 119, 30, 98, 99, 78, 57, 34, 54, 56, 39, 50, 35, 45, 38, 44, 51, 55, 75, 121, 108, 114, 103, 120, 102, 135, 134, 136, 65, 146, 226, 149, 245, 137, 33, 224, 58, 144, 28, 100, 215, 63, 235, 208, 9, 126, 199, 127, 111, 210, 156, 10, 168, 6, 26, 67, 247, 254, 128, 176, 206, 74, 230, 221, 211, 157, 112, 185, 207, 141, 32, 253, 0, 246, 22, 196, 89, 11, 31, 152, 140, 200, 249, 5, 209, 25, 122, 1, 84, 197, 182, 36, 80, 212, 40, 72, 155, 183, 133, 192, 177, 41, 42, 60, 239, 23, 104, 24, 227, 202, 236, 52, 96, 240, 232, 163, 125, 173, 130, 53, 81, 243, 180, 4, 161, 150, 82, 83, 37, 8, 188, 186, 68, 116, 217, 110, 194, 167, 73, 222, 70, 234, 139, 166, 190, 233, 171, 179, 91, 201, 14, 29, 2, 160, 154, 71, 148, 93, 204, 94, 101, 69, 251, 158, 237, 59, 151, 123, 129, 198, 205, 175, 20, 105, 19, 216, 191, 86, 223, 76, 184, 61, 153, 164, 193, 49, 113, 16, 21, 132, 17, 169, 85, 117, 18, 47, 77, 219, 147, 15, 97, 181, 138, 213, 143, 203, 172, 252, 170, 62, 187, 12, 43, 46, 159, 107, 79, 250, 131, 87, 214, 165, 218, 66, 64, 92, 142, 225, 241, 88, 174, 231, 13, 90, 162, 106, 124, 189, 145, 242, 255, 244, 220, 48, 95, 229, 7, 228, 238, 178, 3, 248, 195]
    for i in range(len(strands)):
        strands[i] = bytes([INDEX_X_Y[x] for x in strands[i][:length]]) + strands[i][length:]


def find_interval(lst, num,last_nums=0):
    for i in range(max(0,last_nums-2),len(lst)-1):
        if lst[i] <= num < lst[i+1]:
            return i,lst[i]
    return len(lst)-1,lst[-1]

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

def decode_enhanced_header(header,BLOCK_COUNT):
    header_int = struct.unpack(">L", header)[0]
    data, check = divmod(header_int, 16)
    check_passed = verify_crc4(data<<4, check)
    
    data, length = divmod(data, 1<<4)
    data, part = divmod(data, 1<<5)
    image_id, index = divmod(data, 1<<17)
    outer = False
    if index >= BLOCK_COUNT :
        outer = True
    if not check_passed:
        print(image_id, index, part, length, outer)
    return image_id, index, part, length, outer
class Outer_Code:
    """
    Initialization function to set RS code encoder parameters and internal variables.

    Parameters.
    - fig_nums: number of data blocks, used to handle multiple blocks of data.
    - length: length of the data block, control the number of bits each data block can contain.
    - ecc_nums: number of check bits, used for RS code error correction.
    - mode: Controls the recognition mode of the outer code, if mode=0, the first index of each block is not output when encoding, and is recognized based on speculation when decoding. If mode=1, the first index of each block will be output when encoding, so that it is easier to decode and recognize the sequence position. mode=1 will have better decoding effect.
    """
    def __init__(self,BLOCK_COUNT,fig_nums = 1,length=16,ecc_nums = 31,mode = 1):
        self.rs_code=RSCodec(ecc_nums)
        self.fig_nums = fig_nums
        self.length = length
        self.ecc_nums =ecc_nums
        self.reads_nums = 255 -ecc_nums
        self.block_nums_list = []
        for _ in range(fig_nums):
            self.block_nums_list.append(0)
        self.test_list = []
        self.test_read_list  = []
        self.top_index_dict = {}
        for i in range(self.fig_nums):
            self.top_index_dict[i] = []
        self.mode = mode
        self.BLOCK_COUNT = BLOCK_COUNT

    def calculate_additional_digits(self,num1, num2, total_nums=32):
        if num1 == num2 :
            additional_digits = 32
        elif num2 > num1:
            additional_digits = num2 - num1
        else:
            additional_digits = total_nums - num1 + num2
        return additional_digits

    def encode_unit(self,list):
        """
        Generates an RS matrix of 255 rows x length columns based on the provided sequence of image numbers and number of reads_nums, where the index position of the check digit is null.
        
        Parameters.
        - list: A list containing information about the images, each element is a sequence of bytes.
        
        Returns: rs_matrix
        - rs_matrix: generated RS matrix with data and parity bits, but with the index position of the parity bit unpopulated.
        """
        #First generate an rs matrix with index part
        rs_matrix = np.zeros((255,self.length),dtype=int)

        #Next, put the input sequence into the matrix
        for i,row in enumerate(rs_matrix):
            if i == self.reads_nums or i == len(list):
                break
            row[:] = np.frombuffer(list[i], dtype=np.uint8)
        
        #Next, from the non-index part, for each selected column, a checksum bit is generated and written to the matrix
        for i in range(4,self.length):
            now_info = rs_matrix[:self.reads_nums,i].tolist()

            ecc_info = self.rs_code.encode(now_info)
            rs_matrix[:,i] =np.frombuffer(ecc_info, dtype=np.uint8) 
        
        return rs_matrix
    def encode(self,image_id,input_list):
        """
        Encodes the given image data.

        :param image_id: The ID of the image being encoded.
        :param input_list: A list containing the original image data.
        :return: A list of the encoded data including both the original and error correction codes.
        """

        strands_list = []
        for line in input_list:
            strand = revert_header(line, 16)
            strands_list.append(strand)

        ecc_list = []
        #Calculate how many chunks these sequences are to be divided into
        ecc_block_nums = math.ceil(len(strands_list)/self.reads_nums) 

        if self.mode == 1:
            for j in range(len(strands_list)//32+1):
                line = strands_list[j*32]
                image_index, start, current, length, outer = decode_enhanced_header(line[:4],self.BLOCK_COUNT)
                self.top_index_dict[image_index].append(start)
            test_output = open("Output_code_index","w")
            test_output.write(str(self.top_index_dict))
            test_output.close()
        for i in range(ecc_block_nums):
            if i == ecc_block_nums-1:
                last_nums = 2
            else:
                last_nums = 1
            block_nums = 131071-i
            now_list = strands_list[i*self.reads_nums:i*self.reads_nums+self.reads_nums]
            rs_matrix = self.encode_unit(now_list)
            self.test_list.append(rs_matrix)

            for j in range(self.ecc_nums):
                now_info = rs_matrix[self.reads_nums+j,4:].tolist()
                now_index = list(encode_enhanced_header(image_id, block_nums, j, last_nums))
                now_read = now_index+now_info
                ecc_list.append(bytearray(now_read))
                #print(now_read)
        convert_headers(ecc_list,16)
        convert_headers(strands_list, 16)
        for line in strands_list:
            self.test_read_list.append(list(line))
        #self.test_read_list =strands_list 
        result = strands_list+ecc_list
        test_f = open("test_RS","w")
        test_f.write(str(self.test_list))
        test_f.close()
        return result
    
    def decode(self,id_nums,strands_list):
        result_list = []
        blocks_dict = {} 
        reads_dict = {}  
        for i  in range(id_nums):
            if self.mode == 0:
                blocks_dict[i] = [np.zeros((255,self.length),dtype=int)]
            else:
                blocks_dict[i] = []
                for j in range(math.ceil(len(self.top_index_dict[i])/7)):
                    blocks_dict[i].append(np.zeros((255,self.length),dtype=int))
                #print(len(blocks_dict[i]))
            reads_dict[i] = {"info":[],"ecc":[]}
            result_list.append([])
        #The decoded sequence is first sorted into a list.
        for strand in strands_list:
            strand = revert_header(strand, 16)
            image_index, start, current, length, outer = decode_enhanced_header(strand[:4],self.BLOCK_COUNT)
            if image_index == None:
                #print
                continue
            elif image_index <0 or image_index >=id_nums :
                continue
            if outer != True :
                reads_dict[image_index]["info"].append([start,current, length,strand])
            else:
                reads_dict[image_index]["ecc"].append([131071-start,current, length,strand])

        #Next the sequence of each image is sorted separately by the index of the block
        for d in range(id_nums):
            sorted_info_list = sorted(reads_dict[d]["info"],key = lambda x: (x[0],x[1]))
            sorted_ecc_list = sorted(reads_dict[d]["ecc"],key = lambda x: (x[0],x[1]))



            if self.mode == 0:
                #Next, put the sequence of information into the matrix
                block_index = 0 #This is the index of the RS matrix
                last_matrix_index = -1 #This is the index of the part part of the last read
                info_read_nums = len(sorted_info_list) #Number of reads
                pass_mode = False #Check if the sequence has been excluded
                error_index_list = [] #Index of the error sequence index
                part_index = 0 #What part of the RS matrix is this indexed to

                #The next step is to put the sequence of messages inside the RS matrix
                for i in range(info_read_nums):
                    now_read = sorted_info_list[i]

                    now_block_index  = now_read[0]
                    now_matrix_index = now_read[1]
                    now_info = now_read[3]

                    if pass_mode :
                        pass_mode = False
                        continue
                    if i != info_read_nums-1 :
                        next_matrix_index = sorted_info_list[i+1][1]
                        distance =  self.calculate_additional_digits(now_matrix_index,next_matrix_index)
                        if i-1 not in error_index_list and i != 0:
                            last_matrix_index = sorted_info_list[i-1][1]
                        if distance >16 :
                            if self.calculate_additional_digits(last_matrix_index,next_matrix_index) >16:
                                pass_mode = True
                                error_index_list.append(i+1)
                            else:
                                continue
                        if last_matrix_index <= now_matrix_index:

                            blocks_dict[d][block_index][part_index*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
                        else:
                            if part_index ==6 :
                                block_index += 1 
                                part_index = 0
                                blocks_dict[d].append(np.zeros((255,self.length),dtype=int))
                            else:
                                part_index += 1
                            blocks_dict[d][block_index][part_index*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
            else:

                #Next, put the sequence of information into the matrix
                block_index = 0 
                top_index = 0
                last_matrix_index = 0 
                next_matrix_index = self.top_index_dict[d][top_index+1]
                info_read_nums = len(sorted_info_list) 
                part_index = 0 

                for i in range(info_read_nums):
                    
                    now_read = sorted_info_list[i]
                    now_block_index  = now_read[0]
                    now_matrix_index = now_read[1]
                    now_info = now_read[3] 
                    read_matrix_index,nums  = find_interval(self.top_index_dict[d],now_block_index,last_matrix_index)
                    
                    if now_block_index > nums :
                        block_index = read_matrix_index//7
                        part_index = read_matrix_index%7
                        blocks_dict[d][block_index][part_index*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
                    if now_block_index == nums :
                        if now_matrix_index == 0:
                            block_index = read_matrix_index//7
                            part_index = read_matrix_index%7
                            blocks_dict[d][block_index][part_index*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
                            last_matrix_nums = part_index
                        elif now_matrix_index < 15 : 
                            block_index = read_matrix_index//7
                            part_index = read_matrix_index%7
                            blocks_dict[d][block_index][part_index*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
                        else:
                            block_index = (read_matrix_index-1)//7
                            part_index = (read_matrix_index-1)%7
                            blocks_dict[d][block_index][part_index*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
                    

            #The next step is to put the checksum sequence inside the RS matrix
            ecc_read_nums = len(sorted_ecc_list)
            last_matrix_index = 0 
            pass_mode = False 
            error_index_list = [] 

            for i in range(ecc_read_nums):
                now_read = sorted_ecc_list[i]
                now_block_index =now_read[0]
                now_matrix_index = now_read[1]
                now_end_index= now_read[2]
                now_info = now_read[3]
                if now_block_index >block_index or now_matrix_index >=31:
                    continue
                else:
                    blocks_dict[d][now_block_index][7*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
            
            
            #The following tests are performed on the RS matrix：
            for i in range(len(blocks_dict[d])):
                decode_matrix = blocks_dict[d][i]
                bf_matrix = self.test_list[i]
                k=0
                for j in range(224):
                    #print(i,j,decode_matrix[j], bf_matrix[j])
                    if not np.array_equal(decode_matrix[j], bf_matrix[j]):
                        k = j

            for j in range(len(blocks_dict[d])):
                
                matrix = blocks_dict[d][j]

                for k in range(4,self.length):
                    try:
                        #print("====================")
                        #print(matrix[:,k].tolist())
                        new_info = np.frombuffer(rsc.decode(matrix[:,k].tolist())[0], dtype=np.uint8)
                        #print(new_info)
                        #print(list(rsc.decode(matrix[:,k].tolist())))
                        blocks_dict[d][j][:-self.ecc_nums,k] = new_info
                    except Exception as e:
                        pass

                for k in range(self.reads_nums):
                    if j == len(blocks_dict[d])-1 and np.all(blocks_dict[d][j][k,:] == 0):
                        continue
                    result_list[d].append(bytes(blocks_dict[d][j][k,:].tolist()))

        now_result_list = []
        for i in range(id_nums):
            now_result_list.append([])
        for i in range(len(result_list)):
            for line in result_list[i]:
                l_list = list(line)
                if all(item == 0 for item in l_list):
                    continue
                else:
                    now_result_list[i].append(line)


        return now_result_list

    
"""
rsc = RSCodec(31)

a = [0,0,0,1,1,1]

c = rsc.encode(a)
b = np.frombuffer(rsc.encode(a), dtype=np.uint8)
print(b)
b = [0,11,2,3,1,2,197,108,49,18,139]
print(b)
print(rsc.check(b))
#print(np.frombuffer(rsc.decode(b)[0], dtype=np.uint8))
d = 1
try:
    print(rsc.decode(b)[0])
except Exception as e:
    pass

a = [0,91953,0,1]

#print(decode_enhanced_header(bytearray([63,246,124,17])))
"""