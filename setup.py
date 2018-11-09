from config_local import corpus_path, seg_path, bunch_path, space_path, experiment_corpus_path, experiment_seg_path, experiment_bunch_path, experiment_space_path
from corpus_segment import corpus_segment, experiment_corpus_segment
from corpus_to_bunch import corpus_to_bunch, experiment_corpus_to_bunch
from tfidf import vector_space
from nbayes_predict import predict_result

if __name__ == "__main__":
    # 分词
    corpus_segment(corpus_path, seg_path)
    experiment_corpus_segment(experiment_corpus_path, experiment_seg_path)

    # 构建文本对象
    corpus_to_bunch(bunch_path, seg_path)
    experiment_corpus_to_bunch(experiment_bunch_path, experiment_seg_path)

    # 创建词向量空间实例
    stop_word_path = "./hlt_stop_words.txt"
    vector_space(stop_word_path, bunch_path, space_path)
    vector_space(stop_word_path, experiment_bunch_path, experiment_space_path, space_path)

    # 分类预测
    predict_result(space_path, experiment_space_path)