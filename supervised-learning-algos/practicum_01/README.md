The two files LabelEncoder.py and DictVectorizer.py both describe different coding techniques in dealing with the census data present in the file adult.data.  Specifically, LabelEncoding and DictVectorizing are methods for ennumerating categorical data that is non-numerical.  I have also answered the questions in the lab 1 of the practicum by changing the parameters in each model.  I have found the following:

Altering the value of c for SVC (tuning):
   Changing C to 85 results in greatly improved accuracy for SVC (from .63 to .77).  However, changing C to around 25, also results in good accuracy measures .79.

Altering the value of n_estimators for the tree based classifiers:
   Changing the n_estimators parameter results in slightly different accuracies when dealing with labelencoder and dictvectorizer.  In general the accuracies for labelencoder are higher with fairly low standard deviations, compared to dictvectorizer. For example, @ n_estimators = 30: labelencoder is .857 and dictvectorizer is .837.  Also, in general increasing the n_estimators variable leads to slightly higher accuracies and slightly lower standard deviations.  

Altering the value of cv when running models:
    In general increasing the value of cv (I ran all the models on a value of 3), made everything worse (accuracies and standard deviation).  I guess becuase you are slicing the test data into smaller sizes when you increase cv.

Altering the Logisitic Penalty from l2 to l1:

    This leads to a small increase in mean accuracy in logisitic regression for label encoder.  However, it seems to have greatly improved Dictvectorizer's logisitic regression (w/small increase in std dev.)
