import json
import os

from scipy.special.cython_special import eval_sh_legendre


def create_dir(path):
    """
      根据路径自动判断创建文件或文件夹：
      - 路径包含后缀（如 .txt, .log）则创建文件
      - 无后缀则创建文件夹
      """
    try:
        # 分离路径中的文件名和后缀
        _, ext = os.path.splitext(path)

        if ext:  # 有后缀，视为文件
            # 确保父目录存在
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            # 创建文件（若不存在）
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    pass
                print(f"文件创建成功: {path}")
            else:
                print(f"文件已存在: {path}")

        else:  # 无后缀，视为文件夹
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"文件夹创建成功: {path}")
            else:
                print(f"文件夹已存在: {path}")

    except Exception as e:
        print(f"操作失败: {str(e)}")

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

if __name__ == '__main__':
    create_dir("test")
