#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Companion code for LM error correction codes
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DNA_L_M_code.encoder import Codeword_Strand
from DNA_L_M_code.decoder import Strand_Decode
from DNA_L_M_code.error_test import Error_Type
import time 
def list_decimal_to_base4(input_list):
    output_list = []
    for i in range(len(input_list)):
        decimal = input_list[i]
        base = ""
        for _ in range(4):  
            decimal, remainder = divmod(decimal, 4)
            base =str(remainder) +base
        output_list+=list(map(int,base))
    #print(output_list)
    return np.array(output_list)

def base4to255(msg_array):
    output_list = []
    msg_list = list(msg_array)
    #print(msg_list)
    for i in range(len(msg_list)//4):
        now_msg = msg_list[4*i:4*i+4]
        #print(now_msg)
        if 6 in now_msg :
            output_list.append(666)
        else:
            now_value = now_msg[0]*64+now_msg[1]*16+now_msg[2]*4+now_msg[3]
            output_list.append(now_value)
    return output_list

def encode(msg_list,reads_nums=1):
    """
    LM_code encode
    msg : list, len = 15 ,msg
    reads_nums : int, Number of copies

    Return : list, elem = str 
    """
    output_list = []
    for i in range(len(msg_list)):

        msg = list_decimal_to_base4(msg_list[i])

        LM_enc, LM_code_block_length = list(Codeword_Strand(msg, 8).encode())
        
        #print(len(LM_enc),LM_code_block_length)
        base_map = ["A","T","G","C"]
        LM_enc = [base_map[i] for i in list(LM_enc)]
        output_ = ''.join(LM_enc)
        for i in range(reads_nums):
            output_list.append(output_)
    return output_list

def decode(reads_list, merge_threshold):
    """
    LM_code decode
    msg : list, len = 15 ,elem = str

    Return : list,  elem = bytes_list(0-255)
    """
    cluster = StrandCluster(merge_threshold)
    base_map = {"A":0,"T":1,"G":2,"C":3}
    for read in reads_list:
        read = [base_map[i] for i in read]

        decode_msg = Strand_Decode(read, 8, 8).decode()
        decode_msg_list = base4to255(decode_msg)
        cluster.add(decode_msg_list)

    strands = cluster.get_all()

    return strands


class StrandCluster():
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.index_dict = {} 
        self.stat_dict = {}

    def add(self, strand):
        now_index = tuple(strand[:4])
        if 666 in now_index :
            return
        if now_index not in self.index_dict:
            self.stat_dict[now_index] = 1
            self.index_dict[now_index] = {}
            for i in range(len(strand)):
                self.index_dict[now_index][i] = {}
                if strand[i] != 666 :
                    self.index_dict[now_index][i][strand[i]] = 1
        else:
            self.stat_dict[now_index] += 1
            for i in range(len(strand)):
                if strand[i] != 666:
                    if strand[i] in self.index_dict[now_index][i]:
                        self.index_dict[now_index][i][strand[i]] += 1
                    else:
                        self.index_dict[now_index][i][strand[i]] = 1

    def get_all(self, threshold=None):
        result = []
        if threshold == None:
            threshold = self.threshold
        
        for key in self.index_dict:
            if self.stat_dict[key] < threshold:
                continue
            sum_msg = []
            for i in self.index_dict[key]:
                if self.index_dict[key][i] == {}:
                    sum_msg.append(0)
                else:
                    now_value = max(self.index_dict[key][i],key = self.index_dict[key][i].get)
                    sum_msg.append(now_value)
            result.append(bytes(sum_msg))
        return result
    
def convert_headers(x):
    INDEX_X_Y = [27, 115, 118, 109, 119, 30, 98, 99, 78, 57, 34, 54, 56, 39, 50, 35, 45, 38, 44, 51, 55, 75, 121, 108, 114, 103, 120, 102, 135, 134, 136, 65, 146, 226, 149, 245, 137, 33, 224, 58, 144, 28, 100, 215, 63, 235, 208, 9, 126, 199, 127, 111, 210, 156, 10, 168, 6, 26, 67, 247, 254, 128, 176, 206, 74, 230, 221, 211, 157, 112, 185, 207, 141, 32, 253, 0, 246, 22, 196, 89, 11, 31, 152, 140, 200, 249, 5, 209, 25, 122, 1, 84, 197, 182, 36, 80, 212, 40, 72, 155, 183, 133, 192, 177, 41, 42, 60, 239, 23, 104, 24, 227, 202, 236, 52, 96, 240, 232, 163, 125, 173, 130, 53, 81, 243, 180, 4, 161, 150, 82, 83, 37, 8, 188, 186, 68, 116, 217, 110, 194, 167, 73, 222, 70, 234, 139, 166, 190, 233, 171, 179, 91, 201, 14, 29, 2, 160, 154, 71, 148, 93, 204, 94, 101, 69, 251, 158, 237, 59, 151, 123, 129, 198, 205, 175, 20, 105, 19, 216, 191, 86, 223, 76, 184, 61, 153, 164, 193, 49, 113, 16, 21, 132, 17, 169, 85, 117, 18, 47, 77, 219, 147, 15, 97, 181, 138, 213, 143, 203, 172, 252, 170, 62, 187, 12, 43, 46, 159, 107, 79, 250, 131, 87, 214, 165, 218, 66, 64, 92, 142, 225, 241, 88, 174, 231, 13, 90, 162, 106, 124, 189, 145, 242, 255, 244, 220, 48, 95, 229, 7, 228, 238, 178, 3, 248, 195]
    return  INDEX_X_Y[x]

