import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, cpu_count, freeze_support
import datetime


def preprocess():
    dfs = []
    for i in range(1, 6):
        df = pd.read_csv('./data/train' + str(i) + '.csv', header=None)
        lb = pd.read_csv('./data/label' + str(i) + '.csv', header=None)
        df[13] = lb[0]
        dfs.append(df)
    # 合并训练集
    df = pd.concat(dfs)
    # 打乱训练集
    df = df.sample(frac=1, random_state=np.random.randint(0, 100))
    # 存入磁盘
    df.to_csv('./data/train.csv')


class DecisionTree(object):
    def __init__(self, feature_rate=1.0, max_depth=None, min_leaf_size=1, parallel=False):
        self._x = None
        self._y = None
        self._col_names = None
        self._feature_rate = feature_rate
        self._max_depth = max_depth
        self._min_leaf_size = min_leaf_size
        self._parallel = parallel

        self._split_feature = None
        self._split_point = None
        self._mse_score = None
        self._val = None

        self._left_node = None
        self._right_node = None

    def fit(self, x, y):
        # 训练决策树，参数说明如下
        # x，类型为dataFrame，列数为特征数
        # y，类型为dataFrame或者list，列数为1
        self._x = x
        self._y = y
        if type(y) != np.ndarray:
            self._y = np.array(self._y)
        self._col_names = self._x.columns
        self._val = np.mean(self._y)
        self._mse_score = np.power(self._y-self._val, 2).sum()/len(self._y)

        # 如果到达一定深度，停止分裂
        if (self._max_depth is not None) and self._max_depth < 2:
            return self

        # 随机选取feature_rate*col_nums个属性（避免过拟合），选出其中最优的
        feature_num = int(self._feature_rate*len(self._col_names))
        features = np.random.choice(len(self._col_names), feature_num)
        if self._parallel:
            self.find_split_parallel(features)
        else:
            for feature in features:
                tmp_split_feature, tmp_split_point, tmp_mse_score = self.find_best_split_point(feature)
                if tmp_mse_score < self._mse_score:
                    self._split_feature = tmp_split_feature
                    self._split_point = tmp_split_point
                    self._mse_score = tmp_mse_score

        # 如果是寻找最优属性过程中，没能够完成分裂（原因很多种，可能是min_leaf_size限制、mse无法减小限制等），说明该节点是叶节点
        if (self._split_feature is None) or (self._split_point is None):
            return self

        # 选出最优的分割属性、分割点后分裂节点，递归生成子节点
        left_nodes = np.nonzero(np.array(self._x.iloc[:, self._split_feature]) < self._split_point)[0]
        right_nodes = np.nonzero(np.array(self._x.iloc[:, self._split_feature]) >= self._split_point)[0]
        max_depth = self._max_depth - 1 if self._max_depth is not None else None
        self._left_node = DecisionTree(feature_rate=self._feature_rate, max_depth=max_depth,
                                       min_leaf_size=self._min_leaf_size)
        self._left_node.fit(self._x.iloc[left_nodes], self._y[left_nodes])
        self._right_node = DecisionTree(feature_rate=self._feature_rate, max_depth=max_depth,
                                        min_leaf_size=self._min_leaf_size)
        self._right_node.fit(self._x.iloc[right_nodes], self._y[right_nodes])

        return self

    def find_split_parallel(self, features):
        # 并行化寻找最佳分割点
        workers = cpu_count()
        try:
            workers = min(workers, len(features))
        except Exception:
            print('please input correct n_job')
        pool = Pool(processes=workers)
        result = []
        for feature in features:
            result.append(pool.apply_async(self.find_best_split_point, (feature, )))
        # 关闭进程池，等待子进程退出后，拿出结果
        pool.close()
        pool.join()
        # 由于开启不同进程训练树时会发生tree变量的拷贝（而不是引用），所以最后还需要将结果值赋给trees
        for res in result:
            item = res.get()
            if item[2] < self._mse_score:
                self._split_feature = item[0]
                self._split_point = item[1]
                self._mse_score = item[2]

    def find_best_split_point(self, feature):
        # 根据选定特征feature寻找最佳分割点

        points = list(np.argsort(self._x.iloc[:, feature]))
        start = self._x.iloc[points[self._min_leaf_size-1], feature]

        y_square_sum = np.power(self._y, 2).sum()
        y_sum = self._y.sum()
        y_n = len(self._y)
        left_square_sum = np.power(self._y[0:self._min_leaf_size], 2).sum()
        left_sum = self._y[0:self._min_leaf_size].sum()
        left_n = self._min_leaf_size
        right_square_sum = y_square_sum - left_square_sum
        right_sum = y_sum - left_sum
        right_n = y_n - left_n

        tmp_split_feature = None
        tmp_split_point = None
        tmp_mse_score = self._mse_score
        for i in range(self._min_leaf_size, len(self._y)-self._min_leaf_size+1):
            xi, yi = self._x.iloc[points[i], feature], self._y[points[i]]
            tmp = 0
            if xi == start:
                tmp = 1
            if tmp == 0:
                left_score = self.calculate_mse(left_square_sum, left_sum, left_n)
                right_score = self.calculate_mse(right_square_sum, right_sum, right_n)
                score_after_split = left_score*(left_n/y_n) + right_score*(right_n/y_n)
                # 必须要比当前的mse score小才能分裂，不然就有可能出现y都一样但是仍然分裂的状况，因为此时score是相等的
                if score_after_split < tmp_mse_score:
                    tmp_split_feature = feature
                    tmp_split_point = xi
                    tmp_mse_score = score_after_split
            left_square_sum = left_square_sum + yi ** 2
            left_sum = left_sum + yi
            left_n = left_n + 1
            right_square_sum = right_square_sum - yi ** 2
            right_sum = right_sum - yi
            right_n = right_n - 1
            start = xi
        return tmp_split_feature, tmp_split_point, tmp_mse_score

    def predict(self, x):
        # 预测决策树的参数如下：
        # x，类型为array或者dataFrame，其中的每一项表示一个预测对象的特征
        if type(x) != np.ndarray:
            x = np.array(x)
        return [self.predict_one(one) for one in x]

    def predict_one(self, one):
        if (self._split_feature is None) or (self._split_point is None):
            return self._val
        if one[self._split_feature] < self._split_point:
            return self._left_node.predict_one(one)
        return self._right_node.predict_one(one)

    def __repr__(self):
        attr = 'sample size: {} \t value: {} \t mse score: {}\n'.format(len(self._y), self._val, self._mse_score)
        if (self._split_feature is not None) and (self._split_point is not None):
            attr = attr + 'split feature: {} \t split point: {}\n'.format(self._split_feature, self._split_point)
        return attr

    @staticmethod
    def calculate_mse(y_square_sum, y_sum, y_n):
        return (y_square_sum/y_n) - (y_sum**2)/(y_n**2)


class RandomForest(object):
    def __init__(self, feature_rate=1.0, max_depth=None, min_leaf_size=1, n_job=None, n_tree=1, sample_size=None):
        self._feature_rate = feature_rate
        self._max_depth = max_depth
        self._min_leaf_size = min_leaf_size
        self._n_job = n_job
        self._n_tree = n_tree
        self._sample_size = sample_size

        self._trees = [DecisionTree(self._feature_rate, self._max_depth, self._min_leaf_size) for i in
                       range(self._n_tree)]
        self._x = None
        self._y = None

    def fit(self, x, y):
        # 训练森林，参数说明如下
        # x，类型为dataFrame，列数为特征数
        # y，类型为dataFrame，列数为1
        self._x = x
        self._y = y
        if self._n_job:
            self.fit_tree_parallel()
        else:
            for tree in self._trees:
                self.fit_one_tree(tree, self._trees.index(tree))

    def fit_tree_parallel(self):
        # 并行训练森林中的树
        workers = cpu_count()
        try:
            workers = min(workers, self._n_job)
        except Exception:
            print('please input correct n_job')
        pool = Pool(processes=workers)
        result = []
        for tree in self._trees:
            result.append(pool.apply_async(self.fit_one_tree, (tree, self._trees.index(tree))))
        # 关闭进程池，等待子进程退出后，进行赋值
        pool.close()
        pool.join()
        # 由于开启不同进程训练树时会发生tree变量的拷贝（而不是引用），所以最后还需要将结果值赋给trees
        self._trees = [res.get() for res in result]

    def fit_one_tree(self, tree, index):
        # 训练单颗树的参数如下：
        # tree，单棵树的引用，但被其他进程调用时是拷贝
        # index，树的序号
        start = datetime.datetime.now()
        print('fit the {} th tree'.format(index))
        # 首先进行有放回的随机筛选，选出sample_size个样本
        sample_indexes = np.sort(np.random.choice(len(self._x), self._sample_size))
        # 然后调用DecisionTree.fit进行训练
        tree.fit(self._x.iloc[sample_indexes], self._y.iloc[sample_indexes])
        print('{} th tree fit over, time used: {}'.format(index, (datetime.datetime.now()-start).seconds))
        return tree

    def predict(self, x):
        # 预测决策树的参数如下：
        # x，类型为dataFrame，其中的每一项表示一个预测对象的特征
        all_predictions = [tree.predict(x) for tree in self._trees]
        return np.mean(all_predictions, axis=0)


def r2_score(preds, labels):
    assert len(preds) == len(labels)
    y_mean = np.array(labels).mean()
    SSR = (np.power(np.array(preds) - np.array(labels), 2)).sum()
    SST = (np.power(np.array(labels) - y_mean, 2)).sum()
    print('score', 1-SSR/SST)


if __name__ == '__main__':
    freeze_support()

    # # 测试阶段代码，如果仅仅想要测试程序正确性，可以注释求它代码，然后取消下面代码的注释运行，进行检测
    # data_file = 'C:\\Users\\14735\\神经网络实验课\\data'
    # train_file = open(os.path.join(data_file, 'train1.csv'))
    # label_file = open(os.path.join(data_file, 'label1.csv'))
    # df = pd.read_csv(train_file, header=None)
    # lb = pd.read_csv(label_file, header=None)
    # df[13] = lb[0]
    # print(df)
    #
    # # 测试单颗决策树
    # dt = DecisionTree(feature_rate=0.3, max_depth=32, min_leaf_size=2)
    # dt.fit(df.iloc[0:1000, 0:13], df.iloc[0:1000, 13])
    # print(dt)
    #
    # r2_score(dt.predict(df.iloc[0:1000, 0:13]), df.iloc[0:1000, 13])
    #
    # # 测试随机森林
    # start = datetime.datetime.now()
    # freeze_support()
    # rf = RandomForest(feature_rate=0.3, max_depth=16, min_leaf_size=2, n_job=4, n_tree=16, sample_size=5000)
    # rf.fit(df.iloc[0:10000, 0:13], df.iloc[0:10000, 13])
    # print('all th tree fit over, time used: {}'.format((datetime.datetime.now() - start).seconds))
    # r2_score(rf.predict(df.iloc[0:10000, 0:13]), df.iloc[0:10000, 13])

    # 预处理
    preprocess()

    # 从磁盘读入预处理（打乱顺序）后的训练集
    df = pd.read_csv('./data/train.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    # print(df[0:5])

    # 使用随机森林进行训练并用r2 score评分
    freeze_support()
    rf = RandomForest(feature_rate=0.2, max_depth=32, min_leaf_size=50, n_job=10, n_tree=200, sample_size=2000000)
    rf.fit(df.iloc[:, 0:13], df.iloc[:, 13])
    r2_score(rf.predict(df.iloc[:, 0:13]), df.iloc[:, 13])

    # 使用随机森林预测并写入文件
    preds = []
    for i in range(1, 7):
        test_data = pd.read_csv('./data/test' + str(i) + '.csv', header=None)
        tmp = rf.predict(test_data)
        preds.extend(tmp)

    ids = np.arange(1, len(preds) + 1)
    res = pd.DataFrame({'id': ids, 'Predicted': preds})
    res.to_csv('./sub.csv', index=False)
