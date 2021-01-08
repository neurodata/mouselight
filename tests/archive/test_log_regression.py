# import pytest
# import pandas as pd
# import os
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import matplotlib

# from brainlit.algorithms.regression.log_regression import *

# output_feats = "tests/flat_neighbourhoods_3.csv"
# df_iter = pd.read_csv(output_feats, header=None, index_col=0)
# n_features = df_iter.shape[1] - 1
# X_all = df_iter.iloc[:, :n_features]
# y_all = df_iter.iloc[:, n_features:]
# X_all = StandardScaler().fit_transform(X_all)
# X_sel, X_test, y_sel, y_test = train_test_split(
#     X_all, y_all, test_size=40, random_state=42
# )
# classifiers = [
#     MLPClassifier(hidden_layer_sizes=4, activation="logistic", alpha=1, max_iter=1000),
#     LogisticRegression(max_iter=2000),
#     MLP_LR_NN(X_sel, y_sel, n_features),
# ]
# names = {"MLP-LR": "black", "LR": "blue", "MLP-relu-LR": "red"}


# def test_MLP_LR_NN():
#     model, history = MLP_LR_NN(X_sel, y_sel, n_features)
#     assert len(history.history) != 0


# def test_run_classifiers():
#     f = run_classifiers(
#         X_sel, y_sel, X_test, y_test, classifiers, names, filename="test.csv"
#     )
#     t = open(f, "r")
#     assert t.seek(1)
#     t.close()


# def test_plot_data():
#     fig, ax = plot_data(
#         "test.csv", names, "Accuracy", "Accuracy", "MLP-LR vs LR classification"
#     )
#     assert isinstance(fig, matplotlib.figure.Figure)
#     assert isinstance(ax, matplotlib.axes._subplots.Subplot)
