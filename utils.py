import os


# 忽略Mac的.DS
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


# 保存文件
def save_file(save_path, content):
    with open(save_path, "wb") as fp:
        fp.write(content)


# 读取文件
def read_file(read_path):
    with open(read_path, "rb") as fp:
        content = fp.read()
    return content

