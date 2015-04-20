mport pandas as pd
from sklearn import feature_extraction
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None)
df = df.interpolate()

df = df.convert_objects(convert_numeric=True)

categorical_columns = [1, 3, 5, 6, 7, 8, 9, 13, 14]
for i in categorical_columns:
    df = df[df[i] != '?']

# numerically encode each categorical column
# Documentation here:
# http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/preprocessing/feature_encoding.ipynb
label_encoder = LabelEncoder()
for i in categorical_columns:
    df[i] = label_encoder.fit_transform(df[i])

df = df.dropna()
# Drop na drops missing values
df.head(3)

## Run this line separately ##

feature_data = df[df.columns[0:-1]]
target = df[df.columns[-1]]
print feature_data

## Run this line separately

print "\nDECISION TREE\n"
decision_tree_clf = tree.DecisionTreeClassifier()
scores = cross_val_score(decision_tree_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nRANDOM FOREST\n"
# see what happens as we bump up the number of estimators
random_forest_clf = ensemble.RandomForestClassifier(n_estimators=20)
scores = cross_val_score(random_forest_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nBOOSTED TREES\n"
boosted_clf = ensemble.AdaBoostClassifier(n_estimators=20)
scores = cross_val_score(boosted_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nkNN\n"
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=7)
scores = cross_val_score(knn_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nLogistic\n"
logistic_clf = linear_model.LogisticRegression()
scores = cross_val_score(logistic_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

print "\nSVC\n"
# C is a tuning a parameter (remember it's our error budget for how much slack we give the hyperplane)
# mess around with C and see what happens
support_vector_clf = svm.LinearSVC(C=50)
scores = cross_val_score(support_vector_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

# code that get covariance matrix (a matrix that sees how) correlated two features are.
# with second line we can see that eigenvalues of the eigenvalues of the columns 
# if the eigenvalues of say two columns are 0 

## Run this line separately

# Altering C in SVC
print "\nSVC\n"
# C is a tuning a parameter (remember it's our error budget for how much slack we give the hyperplane)
# mess around with C and see what happens
support_vector_clf = svm.LinearSVC(C=85)
scores = cross_val_score(support_vector_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

## Run this line separately

## Altering n_estimators

print "\nRANDOM FOREST\n"
# see what happens as we bump up the number of estimators
random_forest_clf = ensemble.RandomForestClassifier(n_estimators=50)
scores = cross_val_score(random_forest_clf, feature_data, target, cv=3)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

## Run this line separately

## Altering Penalty for Logistic Regression

print "\nLogistic\n"
logistic_clf = linear_model.LogisticRegression(penalty='l1')
scores = cross_val_score(logistic_clf, feature_data, target, cv=5)
print "Mean: {}".format(scores.mean())
print "Std Dev: {}".format(np.std(scores))

