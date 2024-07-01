import random
import numpy as np
from random import seed, shuffle, randint, choice
rd = random.Random() 
def channel_model_unit(code, pr_dict):
    
    del_num = 0
    ins_num = 0
    sub_num = 0
    pt_num = 0
    unit_list = ["A","T","G","C"]
    af_code = ""
    if rd.random() <= pr_dict["column"]:
        return ""
    else:
        for i in range(len(code)):
            ins_times = 0
            while ins_times < 1:  
                if rd.random() <= pr_dict["pi"]:
                    af_code += random.choice(unit_list)
                    ins_num = ins_num + 1
                else:
                    break
            if rd.random() <= pr_dict["pd"]:
                del_num += 1
                continue
            else:
                pt_num += 1
                if rd.random() <= pr_dict["ps"]:
                    target = choice(list(filter(lambda base: base != code[i], ["A", "C", "G", "T"])))
                    sub_num += 1
                    af_code+=target
                else:
                    af_code+=code[i]
    #print(af_code)
    return af_code

def channel_simulation(dna_reads_list,depth,random_sample = False,pr_dict={"column":0,"pi":0,"pd":0,"ps":0}):

    channel_reads_list = []
    dna_nums = len(dna_reads_list)
    seq_nums = dna_nums*depth

    if random_sample == True:

        for _ in range(seq_nums):
            index = random.randint(0,dna_nums-1)
            now_read = dna_reads_list[index]
            channel_reads_list.append(channel_model_unit(now_read,pr_dict))
        return channel_reads_list
    else:
        for read in dna_reads_list:
            for _ in range(depth):
                channel_reads_list.append(channel_model_unit(read,pr_dict))
        shuffle(channel_reads_list)  
        return channel_reads_list
