# encoding=utf-8

import pickle
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import read_file
from config import bunch_path, space_path


def _read_bunch_obj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


def _write_bunch_obj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):

    stpwrdlst = read_file(stopword_path).splitlines()
    bunch = _read_bunch_obj(bunch_path)

    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})
    if train_tfidf_path is not None:
        trainbunch = _read_bunch_obj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5, vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

    _write_bunch_obj(space_path, tfidfspace)

    print("===================*****====================")
    print("tfidf end")
    print("===================*****====================")


if __name__ == '__main__':
    stopword_path = "./hlt_stop_words.txt"
    vector_space(stopword_path, bunch_path, space_path)
