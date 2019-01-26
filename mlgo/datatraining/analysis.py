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
