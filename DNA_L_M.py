#Encoding and decoding of LM error-correcting codes

import platform
import ctypes
import array

import numpy as np

if platform.system().lower() == "windows":
    lib = ctypes.CDLL('./dna_l_m.dll')
else:
    lib = ctypes.CDLL('./dna_l_m.so')

go_encode = lib.encode
go_encode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), # sequence
    ctypes.c_int64,                 # sequence length
    ctypes.c_int64,                 # msg block length
    ctypes.POINTER(ctypes.c_uint8), # result
    ctypes.c_int64,                 # result length
]

go_decode = lib.decode
go_decode.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), # sequence
    ctypes.c_int64,                 # sequence length
    ctypes.c_int64,                 # msg block length
    ctypes.c_int64,                 # block number
    ctypes.POINTER(ctypes.c_uint8), # result
    ctypes.c_int64,                 # result length
]

def encode(seq, msg_blk_len, num_blocks):
    """
    Encodes the given sequence.

    Parameters:
    seq (bytes): The input sequence to be encoded.
    msg_blk_len (int): The length of each message block.
    num_blocks (int): The number of blocks to be encoded.

    Returns:
    list: The encoded result as a list of integers.
    """
    seq_len = len(seq)
    seq_array = array.array("B", seq)
    seq_ptr = (ctypes.c_uint8 * seq_len).from_buffer(seq_array)

    block_length_w_mark = msg_blk_len + 4 + int(np.ceil(np.log2(2 * msg_blk_len))) + 2
    result_len = block_length_w_mark*num_blocks - 2 
    result_array = array.array("B", [0]*result_len)
    result_ptr = (ctypes.c_uint8 * result_len).from_buffer(result_array)

    go_encode(seq_ptr, seq_len, msg_blk_len, result_ptr, result_len)

    return list(result_array)

def decode(seq, msg_blk_len, num_blocks):
    """
    Decode the input sequence into a list of decoded data.

    Parameters:
    seq: The input byte sequence to be decoded.
    msg_blk_len: The length of each message block.
    num_blocks: The number of message blocks to be decoded.

    Returns:
    A list containing the decoded data.
    """
    seq_len = len(seq)
    seq_array = array.array("B", seq)
    seq_ptr = (ctypes.c_uint8 * seq_len).from_buffer(seq_array)

    result_len = msg_blk_len * num_blocks
    result_array = array.array("B", [0]*result_len)
    result_ptr = (ctypes.c_uint8 * result_len).from_buffer(result_array)

    go_decode(seq_ptr, seq_len, msg_blk_len, num_blocks, result_ptr, result_len)

    return list(result_array)