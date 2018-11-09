# encoding=utf-8
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


from utils import read_file, read_bunch_obj, write_bunch_obj
from config import bunch_path, space_path, experiment_bunch_path, experiment_space_path


def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):

    stpwrdlst = str(read_file(stopword_path), encoding="utf8").splitlines()
    bunch = read_bunch_obj(bunch_path)

    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})

    if train_tfidf_path is not None:
        trainbunch = read_bunch_obj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = CountVectorizer(stop_words=stpwrdlst, max_df=0.5, vocabulary=trainbunch.vocabulary)  # 词频矩阵
        transformer = TfidfTransformer()                                # 权值
        tfidfspace.tdm = transformer.fit_transform(vectorizer.fit_transform(bunch.contents))

    else:
        vectorizer = CountVectorizer(stop_words=stpwrdlst, max_df=0.5)
        transformer = TfidfTransformer()

        tfidfspace.tdm = transformer.fit_transform(vectorizer.fit_transform(bunch.contents))
        tfidfspace.vocabulary = vectorizer.vocabulary_

        # 测试代码
        # contents = bunch.contents[0:10]
        # tfidfspace.tdm = transformer.fit_transform(vectorizer.fit_transform(contents))
        # word = vectorizer.get_feature_names()#获取词袋模型中的所有词语
        # weight = tfidfspace.tdm.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        # for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        #     print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        #     for j in range(len(word)):
        #         print(word[j], weight[i][j])

    write_bunch_obj(space_path, tfidfspace)

    print("===================*****====================")
    print("tfidf end")
    print("===================*****====================")


if __name__ == '__main__':
    stopword_path = "./hlt_stop_words.txt"
    vector_space(stopword_path, bunch_path, space_path)
    vector_space(stopword_path, experiment_bunch_path, experiment_space_path, space_path)
