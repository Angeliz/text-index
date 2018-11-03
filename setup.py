# encoding=utf-8
import jieba
import jieba.analyse
import os

rootdir = './data/training/class1/'
list = os.listdir(rootdir)
print(list)

for i in list:
    f = open(rootdir+i, "r")
    str = f.read()

    # seg_list = jieba.cut(str, cut_all=False)
    # print("Default Mode: " + "/ ".join(seg_list))

    keywords = jieba.analyse.extract_tags(str, topK=5, withWeight=False, allowPOS=())
    print(i)
    print(keywords)

    f.close()
