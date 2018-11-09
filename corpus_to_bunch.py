# encoding=utf-8
import pickle
from sklearn.datasets import base

from utils import read_file, listdir_nohidden
from config import seg_path, bunch_path, experiment_seg_path, experiment_bunch_path


def corpus_to_bunch(bunch_path, seg_path):
    '''
    :param bunch_path: Bunch存储路径
    :param seg_path:  分词后语料库路径
    '''
    seg_class_list = listdir_nohidden(seg_path)
    bunch = base.Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(seg_class_list)

    for seg_class_dir in bunch.target_name:

        seg_class_path = seg_path + "/" + seg_class_dir + "/"
        seg_file_list = listdir_nohidden(seg_class_path)

        for seg_file in seg_file_list:
            seg_full_path = seg_class_path + seg_file
            bunch.label.append(seg_class_dir)
            bunch.filenames.append(seg_file)
            bunch.contents.append(read_file(seg_full_path))

    with open(bunch_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)

    print("===================*****====================")
    print("corpus_to_bunch end")
    print("===================*****====================")


def experiment_corpus_to_bunch(bunch_path, seg_path):
    '''
    :param bunch_path: Bunch存储路径
    :param seg_path:  分词后语料库路径
    '''
    bunch = base.Bunch(target_name=[], label=[], filenames=[], contents=[])

    seg_file_list = listdir_nohidden(seg_path)

    for seg_file in seg_file_list:
        seg_full_path = seg_path + "/" + seg_file
        bunch.filenames.append(seg_file)
        bunch.contents.append(read_file(seg_full_path))

    with open(bunch_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)

    print("===================*****====================")
    print("experiment_corpus_to_bunch end")
    print("===================*****====================")


if __name__ == "__main__":
    corpus_to_bunch(bunch_path, seg_path)
    experiment_corpus_to_bunch(experiment_bunch_path, experiment_seg_path)
