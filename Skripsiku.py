import streamlit as st
import streamlit.components.v1 as components

import streamlit as st
import plotly.figure_factory as ff
import numpy as np

import pandas as pd
from imblearn.datasets import fetch_datasets
from pprint import pprint
# from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
from matplotlib import pyplot as plt
import seaborn as sns
from seaborn import scatterplot
from numpy import where
from collections import Counter
import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')

st.title("Skripsiku")

name = 'pen_digits'

dataset = fetch_datasets()[name]

X = dataset.data
y = dataset.target

df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)

st.subheader('Dataset Name :')
st.write(name)
st.write(df)

cv = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index, in cv.split(X,y):
    # st.write("Train: \n", train_index, "\nValidation:\n", test_index) 
    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]

st.subheader('Splitting Dataset')
st.write(X_train.shape)
st.write(y_train.shape)

st.write(X_test.shape)
st.write(y_test.shape)

# st.write(X_test, y_test)

def report(y_test, y_pred):
    # st.write("Confussion Matrix: \n {}".format(confusion_matrix(y_test, y_pred)))
    st.write("Accuracy: {:.4f}".format(balanced_accuracy_score(y_test, y_pred)))
    st.write("Geometric Mean: {:.4f}".format(geometric_mean_score(y_test, y_pred)))
    # a = y_test
    # b = y_pred
    # return a,b

#     return report


st.subheader('Decision Tree')
dc = DecisionTreeClassifier()
dc.fit(X_train, y_train)
y_pred_dc = dc.predict(X_test)

p = report(y_test, y_pred_dc)

st.subheader('Decision Tree with Bagging')
bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0, n_jobs=1)
bc.fit(X_train, y_train)
y_pred_bc = bc.predict(X_test)

p = report(y_test, y_pred_bc)
# p = report(y_test, y_pred_bc)
# a = p[0]
# d = p[1]

st.subheader('Decision Tree with Adaboost')
bs = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=0)
bs.fit(X_train, y_train)
y_pred_bs = bs.predict(X_test)

p = report(y_test, y_pred_bs)
# p = report(y_test, y_pred_bs)
# b = p[0]
# e = p[1]

st.subheader('Decision Tree with Balanced-Bagging')
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=0)
bbc.fit(X_train, y_train)
y_pred_bbc = bbc.predict(X_test)

report(y_test, y_pred_bbc)
# p = report(y_test, y_pred_bbc)
# c = p[0]
# f = p[1]

# Data = [[a, b, c], 
#     [d, e, f]]

# chart_data = pd.DataFrame(
#    Data,
#    columns=['bagging', 'adaboost', 'balance bagging'])

# st.line_chart(chart_data)
# st.bar_chart(chart_data)



# Add histogram data
# st.write(list(p1))
# x2 = np.random.randn(200)
# st.write(x2)
# x1 = list(p1)
# x2 = list(p2)
# x3 = list(p3)

# Group data together
# hist_data = [x1, x2, x3]

# group_labels = ['bagging', 'adaboost', 'balance bagging']

# Create distplot with custom bin_size
# fig = ff.create_distplot(
#          hist_data, group_labels, bin_size=[.1, .25, .5])

# Plot!
# st.plotly_chart(fig, use_container_width=True)

# arr = np.random.normal(1, 1, size=100)
# st.write(arr)
# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)
# HtmlFile = open("view.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 
# print(source_code)
# components.html(source_code,height=500)

# pprint(df)

# st.write(np.random.randn(20, 3))
