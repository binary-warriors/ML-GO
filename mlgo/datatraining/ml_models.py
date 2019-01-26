'''
This file contains the ML-algorithms used to
operate on the data provided by the user
'''

import pandas as pd
import numpy as np
from flask import current_app
import os
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlgo.models import ResultSet
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlgo.datatraining.feature_selection import select_k_best
from mlgo.datatraining.feature_selection import variance_based

'''
Clean the data
if the target is of string type convert into classes
convert any string features into classes or continuous values
'''


class ML_Models():

    data = ""
    num_cols = 0
    num_rows = 0
    dataset_name = ""

    def __init__(self, data_file):
        filepath = os.path.join(current_app.root_path, 'static/data', data_file)
        data = pd.read_csv(filepath, header=0)
        data.reset_index()
        self.data = data
        self.dataset_name = data_file
        self.num_rows = data.shape[0]
        self.num_cols = data.shape[1]

    def get_labels(self, data):
        df = data
        column_names = list(df)
        df.columns = list(range(0, len(df.columns)))
        features = df.drop(columns=[len(df.columns) -1])
        labels = df.get(len(df.columns) -1)
        features.columns = column_names[:-1]
        labels.columns = column_names[-1]
        return features, labels

    def clean_data(self):
        data = self.data
        data.fillna(data.mean(), inplace=True)
        data.fillna(data.median(), inplace=True)
        data.fillna(data.mode(), inplace=True)
        self.data = data

    def scale_data(self, data, scaler='MinMaxScaler'):
        if scaler is None:
            return data

        if scaler not in ['MinMaxScaler', 'Normalizer', 'Quantile_Transform']:
            scaler = 'MinMaxScaler'

        mmc = MinMaxScaler()
        nm = Normalizer()

        if scaler == 'MinMaxScaler':
            print('In MinMaxScaler')
            mmc.fit(data)
            scaled_data = mmc.transform(data)
            print(type(scaled_data))
            return scaled_data
        elif scaler == 'Normalizer':
            scaled_data_temp = nm.fit(data)
            scaled_data = scaled_data_temp.transform(data)
            return scaled_data
        elif scaler == 'Quantile_Transform':
            return quantile_transform(data, n_quantiles=100, random_state=0)

    def select_features(self, features, labels,  params,  algo="All"):
        print("Algo : ",algo)
        print("run1")
        if algo is 'All' or algo not in ['Variance Based', 'K Best']:
            return features,  features.shape[1]
        print('run2')
        if algo == 'Variance Based':
            print("In variance based")
            try:
                params = float(params)
            except:
                params = 0.0

            if params < 0:
                params = 0.0
            new_features = variance_based(features, labels, params)
            #new_test_f = variance_based(test_f, test_l, params)
            return new_features, new_features.shape[1]
        elif algo == 'K Best':
            print("In k best")
            try:
                params = int(params)
            except:
                params = 10
            new_features = select_k_best(features, labels, params)
            #new_test_f = select_k_best(features, labels, params)
            no_features = new_features.shape[1]
            return new_features,  no_features
        print("End of function")

    def decision_tree(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.3)

        print("\n\n", train.shape, "\n\n", test.shape, "\n\n")

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)
        features_list = []
        #train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features, test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        if criterion != 'gini' or criterion != 'entropy':
            criterion = 'gini'

        if max_depth == '' or max_depth is None:
            max_depth = None
        else:
            max_depth = int(max_depth)

        clf = DTC(criterion=criterion,
                  max_depth=max_depth,
                  min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf)
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = 'Decision Tree'
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs

    def svm(self, c=1.0, kernel='rbf', gamma='auto', max_iter=-1, scaler=None, feature_selection='All',p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.2)

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        features_list = []
        # train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features,
        #                                                                     test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            kernel = 'rbf'

        try:
            c = float(c)
        except:
            c = 1.0

        try:
            gamma = float(gamma)
        except:
            gamma = 'auto'

        try:
            max_iter = int(max_iter)
        except:
            max_iter = -1

        clf = SVC(C=c,
                  kernel=kernel,
                  gamma=gamma,
                  max_iter=max_iter)
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = 'Support Vector Machine'
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs

    def naive_bayes(self, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.2)

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        features_list = []
        # train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features,
        #                                                                     test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        clf = GNB()
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = 'Naive Bayes'
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs

    def knn(self, n_neighbors=5, weights='uniform', algorithm=None, leaf_size=30, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.2)

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        features_list = []
        # train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features,
        #                                                                     test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        if weights not in ['uniform', 'distance']:
            weights = 'uniform'
        if algorithm not in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            algorithm = 'auto'

        try:
            n_neighbors = int(n_neighbors)
        except:
            n_neighbors = 5

        try:
            leaf_size = int(leaf_size)
        except:
            leaf_size = 30

        clf = KNC(n_neighbors=n_neighbors,
                  weights=weights,
                  algorithm=algorithm,
                  leaf_size=leaf_size)

        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = 'KNN'
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs

    def random_forest(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.3)

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        features_list = []
        # train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features,
        #                                                                     test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        if criterion != 'gini' or criterion != 'entropy':
            criterion = 'gini'

        if max_depth == '' or max_depth is None:
            max_depth = None
        else:
            max_depth = int(max_depth)

        clf = RandomForestClassifier(criterion=criterion,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf)
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = 'Random Forest'
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs

    def adaboost(self, n_estimators=50, learning_rate=1, algorithm='SAMME.R', scaler=None, feature_selection='All', p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.3)

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        features_list = []
        # train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features,
        #                                                                     test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        if algorithm not in ['SAMME', 'SAMME.R']:
            algorithm = 'SAMME.R'

        try:
            learning_rate = int(learning_rate)
        except:
            learning_rate = 1

        try:
            n_estimators = int(n_estimators)
        except:
            n_estimators = 50

        if int(n_estimators) < 0:
            n_estimators = 50

        clf = AdaBoostClassifier(n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 algorithm=algorithm)
        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = 'Adaboost'
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs

    def cnn(self, activation='relu', solver='adam', alpha=0.0001, max_iter=200, scaler='MinMaxScaler', feature_selection='All', p=0.0):
        data = self.data
        train, test = train_test_split(data, test_size=0.2)

        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        features_list = []
        # train_features, test_features, features_list = self.select_features(train_features, train_labels, test_features,
        #                                                                     test_labels, p, feature_selection)

        train_features = self.scale_data(train_features, scaler=scaler)
        test_features = self.scale_data(test_features, scaler=scaler)

        train_labels = train_labels.values.reshape(train_features.shape[0], 1)
        test_labels = test_labels.values.reshape(test_features.shape[0], 1)
        print(train_features.shape)
        print(train_labels.shape)
        recon_train = np.hstack((train_features, train_labels))
        recon_test = np.hstack((test_features, test_labels))
        reconstructed_data = np.concatenate((recon_train, recon_test))
        reconstructed_data_df = pd.DataFrame(reconstructed_data, index=None)
        re_features = reconstructed_data_df.drop(columns=[len(reconstructed_data_df.columns) - 1])
        re_labels = reconstructed_data_df.get(len(reconstructed_data_df.columns) - 1)

        selected_feat, num_feat = self.select_features(re_features, re_labels, params=p, algo=feature_selection)
        selected_feat = pd.DataFrame(selected_feat, index=None)
        data = pd.concat([selected_feat, re_labels], axis=1)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_labels = self.get_labels(train)
        test_features, test_labels = self.get_labels(test)

        if activation not in ['identity', 'logistic', 'tanh', 'relu']:
            activation = 'relu'

        if solver not in ['lbfgs', 'sgd', 'adam']:
            solver = 'adam'

        try:
            max_iter = int(max_iter)
        except:
            max_iter = 200

        try:
            alpha = float(alpha)
        except:
            alpha = 0.0001

        clf = MLPClassifier(hidden_layer_sizes=(5, 2),
                            activation=activation,
                            solver=solver,
                            alpha=alpha,
                            max_iter=max_iter)

        clf.fit(train_features, train_labels)
        predictions = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)

        rs = ResultSet()
        rs.algo_name = "CNN"
        rs.dataset_name = self.dataset_name
        rs.accuracy = accuracy
        rs.train_test_split = 0.3
        rs.normalization = scaler
        rs.no_of_features = num_feat
        return rs
