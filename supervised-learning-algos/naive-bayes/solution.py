import csv 
import nltk
import random
import time


def sms_features(message):
    """
    This method returns a dictionary of features that we want out model to be based on.  See return statement to see exactly which 
    features are being included in our model.
    """
    return{
         "contains_poundcurrency": "£" in message,
         "contains_free": "Free" in message or "free" in message,
         "contains_clickhere": "Click Here" in message or "click here" in message or "Click here" in message,
         "contains_dotcomandnet": ".net" in message or ".com" in message,
         "contains_XXX": "xxx" in message or "XXX" in message or "XxX" in message,
         "contains_TXT": "TxT" in message or "txt" in message or "TXT" in message,
    }



def get_feature_sets():
     """
    # Step 1: This reads in the rows from the csv file which look like this:
    ['ham', 'I had askd u a question some hours before. Its answer']
    ['spam', "Win your free tickets to Sunday's game, just text 4589 back to our studio."]

    where the first row is the label; ham, spam
    and the second row is the body of the sms message

    # Step 2: Turn the csv rows into feature dictionaries using `sms_features` function above.

    The output of this function run on the example in Step 1 will look like this:
    [
        ({"contains_poundcurrency": true}, spam), # this corresponds to spam, "Win your free tickets to Sunday's game, just text 4589..
        ({"contains_poundcurrency": false}, ham) # this corresponds to ham, 'I had askd u a question some hours before. Its answer'
    ]

    You can think about this more abstractly as this:
    [
        (feature_dictionary, label), # corresponding to row 0
        ... # corresponding to row 1 ... n
    ]
    """
    f = open('/home/vagrant/repos/datasets/sms_spam_or_ham.csv', 'rb')
    # let's read in the rows from the csv file
    rows = []
    
    for row in csv.reader( f ):
        rows.append( row )

    rows = rows[1:]
    
    output_data = []
    
    #  Currently using half of the dataset, will apply to entire dataset when I am more confident of my feature's accuracy.     
    for row in rows[850:2500]:
        try:
           indicator = row[0]
           feature_dict = sms_features(row[1])
        except:
            pass
        data = (feature_dict, indicator)
        output_data.append(data)
    
    f.close()
    return output_data

def get_training_and_validation_sets(feature_sets):
    
       random.shuffle(feature_sets)
    
       count = len(feature_sets)
    
       slicing_point = int(.20 * count)
    
       training_set = feature_sets[:slicing_point]
    
       validation_set = feature_sets[slicing_point:]
    
       return training_set, validation_set

def run_classification(training_set, validation_set):
    # train the NaiveBayesClassifier on the training_set
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    # let's see how accurate it was
    accuracy = nltk.classify.accuracy(classifier, validation_set)
    print "The accuracy was.... {}".format(accuracy)
    return classifier

def predict(classifier, new_tweet):
    """
    Given a trained classifier and a fresh data point (an SMS),
    this will predict its label, either 0 or 1 (spam or ham).
    """
    return classifier.classify(sms_features(new_tweet))


start_time = time.time()

print "Let's use Naive Bayes!"

our_feature_sets = get_feature_sets()
our_training_set, our_validation_set = get_training_and_validation_sets(our_feature_sets)
print len(get_feature_sets())

print "Now training the classifier and testing the accuracy..."
classifier = run_classification(our_training_set, our_validation_set)

end_time = time.time()
completion_time = end_time - start_time
print "It took {} seconds to run the algorithm".format(completion_time)
print "It took {} seconds to run the algorithm".format(completion_time)
