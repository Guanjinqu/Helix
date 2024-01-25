from reedsolo import RSCodec
import numpy as np
import struct
import math
WIDTH           = 4000
HEIGHT          = 8000
IMAGE_QUALITY   = 60
DUPLICATE       = 1
BLOCK_COUNT     = 125000
DATA_STRAND_GROUP_SIZE  = 32
DATA_LENGTH     = 11
IMAGE_NUMS = 1
def revert_header(strand, length):
    INDEX_Y_X = [75, 90, 155, 253, 126, 86, 56, 249, 132, 47, 54, 80, 214, 235, 153, 202, 190, 193, 197, 177, 175, 191, 77, 108, 110, 88, 57, 0, 41, 154, 5, 81, 73, 37, 10, 15, 94, 131, 17, 13, 97, 104, 105, 215, 18, 16, 216, 198, 246, 188, 14, 19, 114, 122, 11, 20, 12, 9, 39, 168, 106, 184, 212, 44, 227, 31, 226, 58, 135, 164, 143, 158, 98, 141, 64, 21, 182, 199, 8, 219, 95, 123, 129, 130, 91, 195, 180, 222, 232, 79, 236, 151, 228, 160, 162, 247, 115, 203, 6, 7, 42, 163, 27, 25, 109, 176, 238, 218, 23, 3, 138, 51, 69, 189, 24, 1, 136, 196, 2, 4, 26, 22, 89, 170, 239, 119, 48, 50, 61, 171, 121, 221, 192, 101, 29, 28, 30, 36, 205, 145, 83, 72, 229, 207, 40, 241, 32, 201, 159, 34, 128, 169, 82, 185, 157, 99, 53, 68, 166, 217, 156, 127, 237, 118, 186, 224, 146, 140, 55, 194, 211, 149, 209, 120, 233, 174, 62, 103, 252, 150, 125, 204, 93, 100, 183, 70, 134, 213, 133, 240, 147, 179, 102, 187, 139, 255, 78, 92, 172, 49, 84, 152, 112, 208, 161, 173, 63, 71, 46, 87, 52, 67, 96, 206, 223, 43, 178, 137, 225, 200, 245, 66, 142, 181, 38, 230, 33, 111, 250, 248, 65, 234, 117, 148, 144, 45, 113, 167, 251, 107, 116, 231, 242, 124, 244, 35, 76, 59, 254, 85, 220, 165, 210, 74, 60, 243]
    return bytes([INDEX_Y_X[y] for y in strand[:length]]) + strand[length:]
def convert_headers(strands, length):
    INDEX_X_Y = [27, 115, 118, 109, 119, 30, 98, 99, 78, 57, 34, 54, 56, 39, 50, 35, 45, 38, 44, 51, 55, 75, 121, 108, 114, 103, 120, 102, 135, 134, 136, 65, 146, 226, 149, 245, 137, 33, 224, 58, 144, 28, 100, 215, 63, 235, 208, 9, 126, 199, 127, 111, 210, 156, 10, 168, 6, 26, 67, 247, 254, 128, 176, 206, 74, 230, 221, 211, 157, 112, 185, 207, 141, 32, 253, 0, 246, 22, 196, 89, 11, 31, 152, 140, 200, 249, 5, 209, 25, 122, 1, 84, 197, 182, 36, 80, 212, 40, 72, 155, 183, 133, 192, 177, 41, 42, 60, 239, 23, 104, 24, 227, 202, 236, 52, 96, 240, 232, 163, 125, 173, 130, 53, 81, 243, 180, 4, 161, 150, 82, 83, 37, 8, 188, 186, 68, 116, 217, 110, 194, 167, 73, 222, 70, 234, 139, 166, 190, 233, 171, 179, 91, 201, 14, 29, 2, 160, 154, 71, 148, 93, 204, 94, 101, 69, 251, 158, 237, 59, 151, 123, 129, 198, 205, 175, 20, 105, 19, 216, 191, 86, 223, 76, 184, 61, 153, 164, 193, 49, 113, 16, 21, 132, 17, 169, 85, 117, 18, 47, 77, 219, 147, 15, 97, 181, 138, 213, 143, 203, 172, 252, 170, 62, 187, 12, 43, 46, 159, 107, 79, 250, 131, 87, 214, 165, 218, 66, 64, 92, 142, 225, 241, 88, 174, 231, 13, 90, 162, 106, 124, 189, 145, 242, 255, 244, 220, 48, 95, 229, 7, 228, 238, 178, 3, 248, 195]
    for i in range(len(strands)):
        strands[i] = bytes([INDEX_X_Y[x] for x in strands[i][:length]]) + strands[i][length:]
    # for i in range(len(strands)):
    #     strands[i] = bytes([INDEX_X_Y[x] for x in strands[i][:4]]) + strands[i][4:]


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

def decode_enhanced_header(header):
    header_int = struct.unpack(">L", header)[0]
    data, check = divmod(header_int, 16)
    check_passed = verify_crc4(data<<4, check)
    
    data, length = divmod(data, 1<<4)
    data, part = divmod(data, 1<<5)
    image_id, index = divmod(data, 1<<17)
    outer = False
    if index >= BLOCK_COUNT :
        outer = True
    #if not check_passed:
    #    print(image_id, index, part, length, outer)
    return image_id, index, part, length, outer
class Outer_Code:
    def __init__(self,fig_nums = 1,length=16,ecc_nums = 31,mode = 1):
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
        输入图片的编号以及reads_nums数量的序列，返回一个255Xlength的RS矩阵，但是校验位的index部位为空
        """
        #首先生成一个rs矩阵，包含index部分
        rs_matrix = np.zeros((255,self.length),dtype=int)

        #接下来将输入序列放到矩阵里
        for i,row in enumerate(rs_matrix):
            if i == self.reads_nums or i == len(list):
                break
            row[:] = np.frombuffer(list[i], dtype=np.uint8)
        
        #接下来从非index部分，每选取一列，生成一个校验位，并写入到矩阵中
        for i in range(4,self.length):
            now_info = rs_matrix[:self.reads_nums,i].tolist()
            ecc_info = self.rs_code.encode(now_info)
            rs_matrix[:,i] =np.frombuffer(ecc_info, dtype=np.uint8) 
        
        return rs_matrix
    def encode(self,image_id,input_list):
        """
        输入图片id以及这个图片生成的序列list，生成带校验序列的序列list
        """

        strands_list = []
        for line in input_list:
            strand = revert_header(line, 16)
            strands_list.append(strand)

        ecc_list = []
        ecc_block_nums = math.ceil(len(strands_list)/self.reads_nums) #计算这些序列要被分为多少个块

        if self.mode == 1:
            for j in range(len(strands_list)//32+1):
                line = strands_list[j*32]
                image_index, start, current, length, outer = decode_enhanced_header(line[:4])
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
            #接下来依次输出校验序列
            for j in range(self.ecc_nums):
                now_info = rs_matrix[self.reads_nums+j,4:].tolist()

                now_index = list(encode_enhanced_header(image_id, block_nums, j, last_nums))
                now_read = now_index+now_info
                ecc_list.append(bytearray(now_read))
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
        """
        输入图片的数量以及内码后的序列，给出外码后的序列集，注意，我这里解码的时候已经把防聚合物的函数给用了.
        此外输出的外码list为 [[第一个图片的序列]，[第二个图片的序列]……]
        """
        result_list = []
        blocks_dict = {}  #这个是最后解码时候的RS矩阵字典
        reads_dict = {}   #这个是简单的分图片的序列字典
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
        #首先对解码后的序列进行一个排序,然后放到一个list里
        for strand in strands_list:
            strand = revert_header(strand, 16)
            image_index, start, current, length, outer = decode_enhanced_header(strand[:4])
            if image_index == None:
                #print
                continue
            elif image_index <0 or image_index >=id_nums :
                continue
            if outer != True :
                reads_dict[image_index]["info"].append([start,current, length,strand])
            else:
                reads_dict[image_index]["ecc"].append([131071-start,current, length,strand])

        #接下来对每个图片的序列分别按照块的index进行排序
        for d in range(id_nums):
            sorted_info_list = sorted(reads_dict[d]["info"],key = lambda x: (x[0],x[1]))
            sorted_ecc_list = sorted(reads_dict[d]["ecc"],key = lambda x: (x[0],x[1]))

            #for i in range(10):

            #    print(list(sorted_info_list[i][3]))
            #print("排序后的各序列数量",len(sorted_info_list),len(sorted_ecc_list))

            if self.mode == 0:
                #接下来将信息序列放到矩阵里
                block_index = 0 #这是RS矩阵的索引
                last_matrix_index = -1 #这是上一个read的part部分的索引
                info_read_nums = len(sorted_info_list) #这是reads的数量
                pass_mode = False #检验该条序列是否已被排除
                error_index_list = [] #错误序列的索引index
                part_index = 0 #这是RS矩阵中哪一部分的索引

                #接下来是将信息序列放到RS矩阵里面
                for i in range(info_read_nums):
                    now_read = sorted_info_list[i]
                    #print(now_read)
                    now_block_index  = now_read[0]
                    now_matrix_index = now_read[1]
                    now_info = now_read[3]

                    #print(np.frombuffer(now_info, dtype=np.uint8))
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
                            #print(np.frombuffer(now_info, dtype=np.uint8))
                            #print(blocks_dict[i][block_index])

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
                #接下来是带top index的解码方案
                #接下来将信息序列放到矩阵里
                block_index = 0 #这是RS矩阵的索引
                top_index = 0
                last_matrix_index = 0 #这是上一个index表的part部分的索引
                next_matrix_index = self.top_index_dict[d][top_index+1]
                info_read_nums = len(sorted_info_list) #这是reads的数量
                #pass_mode = False #检验该条序列是否已被排除
                #error_index_list = [] #错误序列的索引index
                part_index = 0 #这是RS矩阵中哪一部分的索引
                #接下来是将信息序列放到RS矩阵里面

                last_matrix_nums = 0 #这个记录的是上一个索引的具体图像块索引值

                #test_index_list = []
                for i in range(info_read_nums):
                    
                    now_read = sorted_info_list[i]
                    #print(now_read)
                    now_block_index  = now_read[0]
                    now_matrix_index = now_read[1]
                    now_info = now_read[3] 
                    #test_index_list.append([now_block_index,now_matrix_index])
                    read_matrix_index,nums  = find_interval(self.top_index_dict[d],now_block_index,last_matrix_index)
                    
                    #print(now_block_index,now_matrix_index,last_matrix_nums)
                    
            
                        #raise ValueError ( "This is not a positive number!!" )
                    #下面是test
                    #if list(now_info) ==[  20 , 15, 255 , 97 ,243 ,134 ,  2, 154 ,132, 103,  27 , 27,  27 , 27 , 27 , 74] :
                    #    for z in [38,39]:
                    #        for x in range(224):
                    #            print(blocks_dict[d][z][x])
                    #    print(now_block_index,now_matrix_index,read_matrix_index,nums)
                    #if now_block_index ==41087:
                    #    print(now_block_index,now_matrix_index,read_matrix_index,nums,last_matrix_nums)

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
                    

            #接下来是将校验序列放到RS矩阵里面
            ecc_read_nums = len(sorted_ecc_list)#这是校验序列的数量
            last_matrix_index = 0 #这是上一个read的part部分的索引
            pass_mode = False #检验该条序列是否已被排除
            error_index_list = [] #错误序列的索引index

            for i in range(ecc_read_nums):
                now_read = sorted_ecc_list[i]
                now_block_index =now_read[0]
                now_matrix_index = now_read[1]
                now_end_index= now_read[2]
                now_info = now_read[3]
                if now_block_index >block_index or now_matrix_index >=32:
                    continue
                else:
                    blocks_dict[d][now_block_index][7*32+now_matrix_index,:] = np.frombuffer(now_info, dtype=np.uint8)
            
            
            #test:以下是对RS矩阵进行检验：
            for i in range(len(blocks_dict[d])):
                decode_matrix = blocks_dict[d][i]
                bf_matrix = self.test_list[i]
                k=0
                for j in range(224):
                    #print(i,j,decode_matrix[j], bf_matrix[j])
                    if not np.array_equal(decode_matrix[j], bf_matrix[j]):
                        k = j
                        #print("出现不相等",i,j,decode_matrix[j], bf_matrix[j])
                
                #if not np.array_equal(decode_matrix[k], bf_matrix[k]):
                #    break
                

            #接下来是对RS矩阵进行解码，并输出最终结果
            #print("解码RS块的数量",len(blocks_dict[d]))
            test_g = open("test_get","w")
            test_g.write(str(blocks_dict[d]))
            test_g.close()
            for j in range(len(blocks_dict[d])):
                
                matrix = blocks_dict[d][j]
                #test_matrix = self.test_list[j]
                #if matrix.all() != test_matrix.all():
                #    print("==================",j)
                #    print(matrix)
                #    print(test_matrix)
                #if j == 0 :
                    #print("```````````````````")
                    #print(matrix[:,1].tolist())
                    #print(test_matrix[:,1].tolist())
                    #print("```````````````````")
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
                        #print("出现报错：",d,j,k,len(blocks_dict[d]))
                        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                for k in range(self.reads_nums):
                    if j == len(blocks_dict[d])-1 and np.all(blocks_dict[d][j][k,:] == 0):
                        continue
                    result_list[d].append(bytes(blocks_dict[d][j][k,:].tolist()))
        print("外码后输出的序列",len(result_list[0]))

        return result_list

    
