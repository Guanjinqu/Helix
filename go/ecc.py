#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import DNA_L_M
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

        LM_enc = list(DNA_L_M.encode(msg, 4, 15))
        
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

        decode_msg = DNA_L_M.decode(read, 4, 15)
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