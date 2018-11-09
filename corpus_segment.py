# encoding=utf-8
import jieba
import jieba.analyse
import os

from utils import listdir_nohidden, save_file, read_file
from config_local import corpus_path, seg_path, experiment_corpus_path, experiment_seg_path


# 分词
def corpus_segment(corpus_path, seg_path):
    '''
    :param corpus_path: 未分词语料库路径
    :param seg_path: 分词后语料库存储路径
    '''

    class_list = listdir_nohidden(corpus_path)

    for class_dir in class_list:
        class_path = corpus_path + "/" + class_dir + "/"
        seg_class_path = seg_path + "/" + class_dir + "/"

        if not os.path.exists(seg_class_path):
            os.makedirs(seg_class_path)

        file_list = listdir_nohidden(class_path)

        for file in file_list:
            full_path = class_path + file
            content = read_file(full_path)
            content_seg = jieba.cut(content)
            save_file(seg_class_path + file, bytes(" ".join(content_seg), encoding="utf8"))  # 将处理后的文件保存到分词后语料目录

    print("===================*****====================")
    print("corpus_segment end")
    print("===================*****====================")


def experiment_corpus_segment(corpus_path, seg_path):
    '''
    :param corpus_path: 未分词语料库路径
    :param seg_path: 分词后语料库存储路径
    '''

    file_list = listdir_nohidden(corpus_path)

    seg_path0 = seg_path + "/"
    if not os.path.exists(seg_path0):
        os.makedirs(seg_path0)

    for file in file_list:
        full_path = corpus_path + "/" + file
        content = read_file(full_path)
        content_seg = jieba.cut(content)
        save_file(seg_path0 + file, bytes(" ".join(content_seg), encoding="utf8"))  # 将处理后的文件保存到分词后语料目录

    print("===================*****====================")
    print("experiment_corpus_segment end")
    print("===================*****====================")


if __name__ == "__main__":
    # 对训练集进行分词
    corpus_segment(corpus_path, seg_path)

    # 对测试集进行分词
    experiment_corpus_segment(experiment_corpus_path, experiment_seg_path)
