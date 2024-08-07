import random
import numpy as np
from random import seed, shuffle, randint, choice
rd = random.Random(2) #该步骤是防止受随机种子的影响 
def channel_model_unit(code, pr_dict):
    """
    Channel simulation function.
    This function simulates deletion, insertion, and substitution errors that may occur as the signal passes through the channel.
    The probability of an error occurring is determined by the value in the pr_dict dictionary.
    
    Parameters.
    code (str): the input signal sequence.
    pr_dict (dict): a dictionary containing the probability of various errors occurring, with keys 'column', 'pi', 'pd' and 'ps'.
    
    Returns.
    str: sequence of signals that may have been modified after transmission through the channel.
    """
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
    """
    Simulates the process of DNA sequence passing through a sequencing channel.

    Parameters:
    dna_reads_list: A list of DNA read sequences.
    depth: Sequencing depth, representing the number of times each DNA molecule is sequenced.
    random_sample: Whether to perform random sampling during sequencing simulation. Default is False.
    pr_dict: A dictionary containing channel error rates, including substitution, insertion, deletion, and stay probabilities. Default is all 0.

    Returns:
    A list of sequences after passing through the channel simulation.
    """
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