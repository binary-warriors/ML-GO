from flask import current_app
import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency


class Analysis:

    data = ""
    num_cols = 0
    num_rows = 0
    dataset_name = ""

    def __init__(self, data_file):
        filepath = os.path.join(current_app.root_path, 'static/data', data_file)
        data = pd.read_csv(filepath, header=0)
        if data.shape[1] == 1:
            data = pd.read_csv(filepath, header=0, delimiter='\t')
        data.reset_index()
        self.data = data
        columns = data.columns
        for col in columns[:-1]:
            try:
                data[col] = data[col].astype('float64')
            except:
                data[col] = pd.factorize(data[col])[0]
        try:
            data[columns[-1]] = data[columns[-1]].astype('int64')
        except:
            data[columns[-1]] = pd.factorize(data[columns[-1]])[0]

        data.fillna(data.mean(), inplace=True)
        data.fillna(data.median(), inplace=True)
        data.fillna(data.mode(), inplace=True)
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

    def pca(self):
        features, targets = self.get_labels(self.data)
        model = PCA()
        model.fit(features, targets)
        covariance = model.get_covariance()
        precision = model.get_precision()
        score = model.score(features, targets)
        log_likelihood = model.score_samples(features)
        cv = ""
        prec = ""
        for line in covariance:
            for point in line:
                cv = cv + "<span style=\"padding-right:1em\"></span>" + str(np.round(point, decimals=2))
            cv = cv + "<br>"

        for line in precision:
            for point in line:
                prec = prec + "<span style=\"padding-right:1em\"></span>" + str(np.round(point, decimals=2))
            prec = prec + "<br>"

        result_dict = {
            'Test': 'Principal Component Analysis',
            'Score': score,
            'Covariance': cv,
            'Precision': prec
            # 'Log-Likelihood': log_likelihood
        }
        return result_dict

    def chi2(self):
        features, targets = self.get_labels(self.data)
        chi2_stat, p_value, degree_of_freedom, _not_used = chi2_contingency(features)
        result_dict = {
            'Test': 'Chi-Squared Analysis',
            'Chi2Stat': chi2_stat,
            'P value': p_value,
            'Degree of Freedom': degree_of_freedom
        }
        return result_dict
