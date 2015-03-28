### Run this code separately: Decision Tree ###
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier 
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


df = pd.read_csv('https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/iris.csv')

decision_tree_clf = tree.DecisionTreeClassifier(
    # dont split if the max number of samples on either side of the slice is less than 2
    min_samples_leaf=2)
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# use K-fold cross validation, with k=5 and get a list of accuracies
scores = cross_val_score(decision_tree_clf, df[features], df["target"], cv=5)

print "\nDECISION TREE:\n"
# print the accuracies
print scores
# print the average accuracy
print scores.mean()
# print the standard deviation of the scores
print np.std(scores)


# now let's try a bagging example
bagging_clf = BaggingClassifier(
    decision_tree_clf,
    #you can pass other classifiers here.
    # bag using 20 trees
    n_estimators=20,
    # the max number of samples to draw from the training set for each tree
    # there are 105 training samples, so each tree will have .8 * 105
    # data points each to train on, chosen randomly with replacement
    max_samples=0.8,
)


# use K-fold cross validation, with k=5 and get a list of accuracies
# this separates the data into training and test set 5 different times for us
# and finds out the accuracy in each case to get a sense of the average accuracy
scores = cross_val_score(bagging_clf, df[features], df["target"], cv=5)

print "\nDECISION TREE WITH BAGGING:\n"
# print the accuracies
print scores
# print the average accuracy
print scores.mean()
# print the standard deviation of the scores
print np.std(scores)

### Run this code separately: KNN###
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier 
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/iris.csv')

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

clf = KNeighborsClassifier(n_neighbors=5)
score2 = cross_val_score(clf, df[features], df["target"], cv=5)

print score2
# print the average accuracy
print score2.mean()
# print the standard deviation of the scores
print np.std(score2)

### Run this code separately: KNN w/bagging ###

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier 
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/iris.csv')

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

clf = KNeighborsClassifier(n_neighbors=5)


bagging_clf = BaggingClassifier(
    clf,
    #you can pass other classifiers here.
    # bag using 20 trees
    n_estimators=20,
    # the max number of samples to draw from the training set for each tree
    # there are 105 training samples, so each tree will have .8 * 105
    # data points each to train on, chosen randomly with replacement
    max_samples=0.8,
)

score2 = cross_val_score(bagging_clf, df[features], df["target"], cv=5)

print score2
# print the average accuracy
print score2.mean()
# print the standard deviation of the scores
print np.std(score2)

### Run this code separately: Logisitc Regression ###

import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier 
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


df = pd.read_csv('https://s3-us-west-2.amazonaws.com/ga-dat-2015-suneel/datasets/iris.csv')

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

logistic_clf = linear_model.LogisticRegression()
scores = cross_val_score(logistic_clf, df[features], df["target"], cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))


