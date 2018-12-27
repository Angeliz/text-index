# encoding=utf-8
from sklearn.neighbors import KNeighborsClassifier

from utils import read_bunch_obj
from config_local import space_path, experiment_space_path


def predict_result(space_path, experiment_space_path):

    train_set = read_bunch_obj(space_path)
    test_set = read_bunch_obj(experiment_space_path)

    # 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高
    clf = KNeighborsClassifier(alpha=0.001).fit(train_set.tdm, train_set.label)

    # 预测分类结果
    predicted = clf.predict(test_set.tdm)

    for file_name, expct_cate in zip(test_set.filenames, predicted):
        print(file_name, ": 实际类别:", "---", " -->预测类别:", expct_cate)

    print("===================*****====================")
    print("predict end")
    print("===================*****====================")


# # 计算分类精度：
# def metrics_result(actual, predict):
#     print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
#     print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
#     print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
#
#
# metrics_result(test_set.label, predicted)


if __name__ == '__main__':
    predict_result(space_path, experiment_space_path)