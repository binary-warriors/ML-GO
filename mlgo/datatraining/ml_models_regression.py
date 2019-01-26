from flask import current_app
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import quantile_transform
from sklearn.preprocessing import Normalizer
from mlgo.datatraining.feature_selection import select_k_best
from mlgo.datatraining.feature_selection import variance_based
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from mlgo.models import ResultSetRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

class RegressionModels:
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

    def get_targets(self, data):
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

    def sgd(self, loss='squared_loss', penalty='l2', alpha=0.0001, max_iter=1000, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        data = self.scale_data(data=data, scaler=scaler)

        data = pd.DataFrame(data, index=None)

        features, targets = self.get_targets(data)
        selected_features, num = self.select_features(features, targets, params=p, algo=feature_selection)

        targets = targets.values.reshape(selected_features.shape[0], 1)
        data = np.hstack((selected_features, targets))
        data = pd.DataFrame(data, index=None)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_targets = self.get_targets(train)
        test_features, test_targets = self.get_targets(test)

        if loss not in ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
            loss = 'squared_loss'

        if penalty not in ['none', 'l2', 'l1', 'elasticnet']:
            penalty = 'l2'

        try:
            alpha = float(alpha)
        except:
            alpha = 0.0001

        try:
            max_iter = int(max_iter)
        except:
            max_iter = 1000


        reg = SGDRegressor(loss=loss,
                           penalty=penalty,
                           alpha=alpha,
                           max_iter=max_iter)
        model = reg.fit(train_features, train_targets)
        predictions = model.predict(test_features)
        rms = mean_squared_error(test_targets, predictions)

        rs = ResultSetRegressor()
        rs.algo_name = 'SGD'
        rs.dataset_name = self.dataset_name
        rs.rms = rms
        rs.no_of_features = num
        rs.train_test_split = 0.3
        rs.normalization = scaler
        return rs

    def lasso(self, alpha=1.0, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        data = self.scale_data(data=data, scaler=scaler)

        data = pd.DataFrame(data, index=None)

        features, targets = self.get_targets(data)
        selected_features, num = self.select_features(features, targets, params=p, algo=feature_selection)

        targets = targets.values.reshape(selected_features.shape[0], 1)
        data = np.hstack((selected_features, targets))
        data = pd.DataFrame(data, index=None)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_targets = self.get_targets(train)
        test_features, test_targets = self.get_targets(test)

        try:
            alpha = float(alpha)
        except:
            alpha = 1.0

        reg = Lasso(alpha=alpha)
        model = reg.fit(train_features, train_targets)
        predictions = model.predict(test_features)
        rms = mean_squared_error(test_targets, predictions)

        rs = ResultSetRegressor()
        rs.algo_name = 'Lasso Regression'
        rs.dataset_name = self.dataset_name
        rs.rms = rms
        rs.no_of_features = num
        rs.train_test_split = 0.3
        rs.normalization = scaler
        return rs

    def linear(self, scaler=None, feature_selection='All', p=0.0):
        data = self.data
        data = self.scale_data(data=data, scaler=scaler)

        data = pd.DataFrame(data, index=None)

        features, targets = self.get_targets(data)
        selected_features, num = self.select_features(features, targets, params=p, algo=feature_selection)

        targets = targets.values.reshape(selected_features.shape[0], 1)
        data = np.hstack((selected_features, targets))
        data = pd.DataFrame(data, index=None)

        train, test = train_test_split(data, test_size=0.3)
        train_features, train_targets = self.get_targets(train)
        test_features, test_targets = self.get_targets(test)

        reg = LinearRegression()
        model = reg.fit(train_features, train_targets)
        predictions = model.predict(test_features)
        rms = mean_squared_error(test_targets, predictions)

        rs = ResultSetRegressor()
        rs.algo_name = 'Linear Regression'
        rs.dataset_name = self.dataset_name
        rs.rms = rms
        rs.no_of_features = num
        rs.train_test_split = 0.3
        rs.normalization = scaler
        return rs
