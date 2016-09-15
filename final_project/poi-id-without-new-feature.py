# -*- coding: utf-8 -*-
'''
Created on Tue Aug 30 14:52:52 2016

@author: mb4570
'''
#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")  
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cross_validation import StratifiedShuffleSplit
from pprint import pprint
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import *
from sklearn import metrics
from sklearn.feature_selection import *
from tester import test_classifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 
                 'deferred_income','total_stock_value', 'exercised_stock_options', 
                'restricted_stock', 'restricted_stock_deferred', 'expenses',  
                 'long_term_incentive', 'shared_receipt_with_poi', 
                 'from_this_person_to_poi','from_poi_to_this_person',
                'to_messages','from_messages'] 
                

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# List of all keys of the data_dict for  salary value > 1 million and 
# bonus > 5 million dollars
outliers = []
for e in data_dict.keys():
    if data_dict[e]["salary"] != 'NaN' and data_dict[e]['salary'] > 1000000 and data_dict[e]['bonus'] > 5000000:
        outliers.append(e)
        
print "Outliers Before Removal :",outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = data_dict

# Cleaning up manually my_dataset by removing records with ave_earnings=0 because 
#it indicates the entries that has no financial data, would not have valuable input for recognizing POI


        
print "\nTotal number of data points:", len(my_dataset)

''' 
#### I will add a new feature that shows average value of total earning called ave_earnings 
#### by calculating the mean value of 'salary', 'bonus', 'deferral_payments', and 'total_payments' for each person.

for ele in my_dataset:
    earnings = []
    for e in features_list[1:5]:
        earn = my_dataset[ele][e]
        if earn =='NaN':
            earn = 0
            earnings.append(earn)
        earnings.append(earn)
    ave_earnings = np.mean(earnings)
    my_dataset[ele].update({'ave_earnings': ave_earnings})

print 'ave_earnings is the average value of:', features_list[1:5]

print "List of Employees with no Financial records and their POI status:\n"
for ele in data_dict:
    if my_dataset[ele]['ave_earnings']==0:
        print ("Name: " + str(ele) + ", POI: " + str(my_dataset[ele]['poi'])) 

# Since all Employees with no financial records are non-POI, we can remove them
# from the dataset in order to increase the chance of recognizing POIs.

my_dataset .pop('CORDES WILLIAM R',0)
my_dataset .pop('LOWRY CHARLES P',0)
my_dataset .pop('CHAN RONNIE',0)
my_dataset .pop('BELFER ROBERT',0)
my_dataset .pop('WHALEY DAVID A',0)
my_dataset .pop('CLINE KENNETH W',0)
my_dataset .pop('LEWIS RICHARD',0)
my_dataset .pop('MCCARTY DANNY J',0)
my_dataset .pop('POWERS WILLIAM',0)
my_dataset .pop('PIRO JIM',0)
my_dataset .pop('WROBEL BRUCE',0)
my_dataset .pop('MCDONALD REBECCA',0)
my_dataset .pop('SCRIMSHAW MATTHEW',0)
my_dataset .pop('GATHMANN WILLIAM D',0)
my_dataset .pop('GILLIS JOHN',0)
my_dataset .pop('MORAN MICHAEL P',0)
my_dataset .pop('LOCKHART EUGENE E',0)
my_dataset .pop('SHERRICK JEFFREY B',0)
my_dataset .pop('FOWLER PEGGY',0)
my_dataset .pop('CHRISTODOULOU DIOMEDES',0)
my_dataset .pop('HUGHES JAMES A',0)
my_dataset .pop('HAYSLETT RODERICK J',0)

'''
###Extract features and labels from dataset for local testing

# I removed entries with all '0' by utilizing the featurFormat parameters 
# in order to clean up the data and avoid any problem on score calcultions. 
# Especially when these data entries would not have any impact on finding POI.

data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True,
                     remove_all_zeroes=True, remove_any_zeroes=False)
""" convert my_dataset to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = False will not omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.                    
"""  
      
labels, features = targetFeatureSplit(data)


# Statistical Overview of the data:

print "\nTotal number of data points in data_dict:", len(data_dict) 
print "\nTotal number of data points in my_dataset:", len(my_dataset) 
print "\n Original Features List:\n"
pprint (features_list)
print "\nNumber of features:", len(features_list)
print "Length of my_dataset before removing all zeros=",len(my_dataset)   
print "Length of data after removing all zeros =",len(data)  
i=0
j=0
for ele in my_dataset:
    if my_dataset[ele]['poi']==True:
        i+=1
    else:
        j+=1

print "Total Number of POI:", i
print "Total Number of Non-POI:", j


### Task 4: Try a varity of classifiers

# In order to simplify the calculation of the clasifiers scores I created the
# following function that will calculate score repors output of the tester.py 
# and also sklearn.metrics.classification_report for comparison

def tester_scores ():
    print ' '
    print "Tester Classifier Report" 
    test_classifier(clf, my_dataset, features_list)
    print ' '

    
print ' '
print "varity of Clasifiers Verification Before any fine tuning:"

# I tried several other classifiers and their output results of the 
# tester.py script can be found in the ML_project_varity_of_classifiers.ipny file.
# Below is the results for following four classifiers that I will compare them 
# through different methods of analysis.
# 1. Logistic Regression
# 2. LinearSVC
# 3. Decision Trees
# 4. Random Forest
# 5. Gaussian NB 
 

clf = GaussianNB()
tester_scores ()

# None of the clasifiers returning precision and recall scores above .3

### Task 5: Tune your classifier to achieve better than .3 precision and recall 

# I make further modifications on the features_list by applying MinMaxScaler and
# SelectKBest based on Chi-squared scoring function to choose 10 best features. 
   
scalar = MinMaxScaler()
scaled_features = scalar.fit_transform(features)

#print scaled_features 

features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled_features, labels, test_size=0.1, random_state=42)

# Manually tried several k values, Number of top features to select, for Chi-squared the k=10 was returning best 
# results for different methods and clasifiers. 

chi2 = SelectKBest(chi2, 10)
features_train = chi2.fit_transform(features_train, labels_train)
features_test = chi2.transform(features_test)

# keep selected feature names
# i+1 because we still have poi as the first name in the feature_list, while the actual features matrix does not

features_list_new = [features_list[i+1] for i in chi2.get_support(indices=True)]

features_list = ["poi"] + features_list_new
print "chi2 selected features_list = "
pprint (features_list)

# I will apply featureFormat to new feature_list with 10 best members and extraxt 
# new labels/features to use them for the same varity of clasifiers and compare their scores.

data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)


clf = GaussianNB()
tester_scores () 

# Except for LR() other clasifiers show improved scores and only Decision tree 
# clasifier has around .3 precision and recall values. 

# Next I will apply GridSearchCV method with stratified shuffle split cross validation  
# to get the best parametr sets for each clasifier after applying it in a pipeline of 
# SelectKbest, pca to send the principal components of selected features to the clasifier.

def gridsearchcv(clfi):
    sk_fold = StratifiedShuffleSplit(labels, 100, test_size=0.1, random_state = 42)
    skb = SelectKBest()
    pca = PCA()
    clf = clfi

    pipe = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("CLF", clf)])


    pca_params = {"PCA__n_components":range(1,4), "PCA__whiten": [True, False]}
    kbest_params = {"SKB__k":range(4,10)} 
    pca_params.update(kbest_params)

    gs =  GridSearchCV(pipe, pca_params,   scoring='f1',  cv=sk_fold)
# Fit GridSearchCV
    gs.fit(features, labels)

    clf = gs.best_estimator_
    return clf
    

clf = gridsearchcv(GaussianNB())
tester_scores() 

#The best results above .3 for precision and recall is for GaussianNB(). So I will 
# use this clasifier in the pipeline defined in gridsearchcv function:

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)

