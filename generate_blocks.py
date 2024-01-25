import os
from collections import Counter
import pickle

import coder

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

counter = Counter()

def generate_file():
    for i in findAllFile("./test_images"):
        print(i)
        if not i.endswith(".bmp"):
            continue
        raw_data = coder.image_encoding(i, coder.IMAGE_QUALITY)
        blocks = coder.get_image_blocks(raw_data)
        counter.update(blocks)
        # break
    most_common_256_blocks = counter.most_common(65536)
    block2index = {}
    index2block = []
    for i, block in enumerate(most_common_256_blocks):
        block2index[block[0]] = i
        index2block.append(block[0])

    hardcoded_data = {
        "block2index": block2index,
        "index2block": index2block
    }
    f = open("data_65536.pkl","wb")
    pickle.dump(hardcoded_data, f)
    f.close()

def load_data():
    f = open("data_65536.pkl","rb")
    hardcoded_data = pickle.load(f)
    f.close()
    return hardcoded_data

generate_file()
print(load_data())