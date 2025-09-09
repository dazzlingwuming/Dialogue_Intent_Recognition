import json
import os

from scipy.special.cython_special import eval_sh_legendre


def create_dir(file_path):
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        pass

#读取文件
def read_file(file_path):
    if isinstance(file_path, str):
        files =[file_path]
    #判断是不是json文件
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                yield item
    else:
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

