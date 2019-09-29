"""
Classification Pipeline
"""
from sklearn import preprocessing, decomposition, svm, cross_validation
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import *
import random
import numpy as np
import matplotlib.pylab as pl
import pandas as pd



#####################################################################################
# Classifiers and their initialization parameters in the dictionary below: # 
# The runClassification function will accept a list with the classifiers that the   #
# we wish to run.                                                                   #
#####################################################################################

# TODO: Abstract these following classifiers so they can be passed into runClassification with their own parameters
clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier()
        }

attributes = {'study_hrs': 0, 'travel_time': 6, 'course_id': 21, 'attendance': 22, 'challenges_family': 13, 'mother_occup': 12, 'health': 1, 'test': 23, 'father_occup': 11, 'hsc': 18, 'medium': 19, 'campus_feedback': 5, 'mother_tongue': 15, 'mother_edu': 10, 'family_type': 7, 'source_fees': 3, 'tuition': 2, 'backlogs': 16, 'father_edu': 9, 'ssc': 17, 'drop_year': 4, 'annual_income': 8, 'gender': 20, 'caste': 14}

attribute = {0: ['0','1','2','3','4','5'],
1: ['EXCELLENT','POOR','AVERAGE','GOOD'],
2: ['NO','YES'],
3: ['PARENTS','SCHOLARSHIP','MULTIPLE','RELATIVES'],
4: ['NO','YES'],
5: ['1','2','3','4','5'],
6: ['UPTO 1 HOUR','1 TO 2 HOURS','2 TO 3 HOURS','MORE THAN 3 HOURS'],
7: ['NUCLEAR','JOINT','EXTENDED'],
8: ['LESS THAN 1 LAKH','1 TO 1.99 LAKHS','2 TO 2.99 LAKHS','3 TO 3.99 LAKHS','4 TO 4.99 LAKHS','5 LAKHS AND ABOVE'],
9: ['0','1','2','3','4','5'],
10: ['0','1','2','3','4'],
11: ['SALARIED EMPLOYEE','BUSINESS','DEFENCE SERVICE(AIRFORCE)','NO JOB','SERVICE','PASSED AWAY','HELPER,DRIVER','SELF EMPLOYEE','ELECTRICIAN','BMC SERVICE (ASST. ENGINEER)'],
12: ['HOUSEWIFE','SALARIED EMPLOYEE','BUSINESS'],
13: ['NONE', 'ANY ILL PERSON', 'SALARIED EMPLOYEE', 'FINANCES'],
14: ['OPEN','OBC','GENERAL','MUSLIM','HINDU'],
15: ['HINDI','URDU','MARATHI','GUJARATI','KONKANI','TAMIL','BENGALI'],
16: ['0','1','2','3','4','5','6','7'],
17: ['DISTINCTION','FIRST CLASS','SECOND CLASS'],
18: ['DISTINCTION','FIRST CLASS','SECOND CLASS'],
19: ['ENGLISH','VERNACULAR'],
20: ['MALE','FEMALE'],
21: ['CPC501', 'CPC502', 'CPC503', 'CPC504', 'CPL501', 'CPL502', 'CSC401', 'CSC402', 'CSC403', 'CSC404', 'CSC405', 'CSC406', 'CSC301', 'CSC302', 'CSC303', 'CSC304', 'CSC305', 'CSC306', 'CPC601', 'CPC602', 'CPC603', 'CPC604', 'CPL605',  'CPE6012'],
22: ['EXCELLENT','GOOD','AVERAGE','POOR'],
23: ['-1','0','4','5','6','7','8','9','10']}

course_id = ['CPC501', 'CPC502', 'CPC503', 'CPC504', 'CPL501', 'CPL502', 'CSC401', 'CSC402', 'CSC403', 'CSC404', 'CSC405', 'CSC406', 'CSC301', 'CSC302', 'CSC303', 'CSC304', 'CSC305', 'CSC306', 'CPC601', 'CPC602', 'CPC603', 'CPC604', 'CPL605',  'CPE6012']

class Model:
    
    def __init__(self, dataSet, dependentVar, doFeatureSelection=True, doPCA=False, nComponents=10):
	
        """ Data pre-processing constructor.

        Constructor to pre-process pandas DataFrames, extracting and encoding the outcome
        labels (class), dropping them from the dataset and converting categorical variables
        into integer numbers for compatibility with scikit-learn.

        Parameters
        ----------
        dataSet : pd.DataFrame
            The entire dataset as loaded and parsed in the main program
        dependentVar : string
            A string denoting the column to be used as the class
        doFeatureSelection : bool
            A flag to denote whether or not to perform feature selection
        doPCA : bool
            A flag to denote whether or not to perform principle component analysis
        nComponents : int
            The desired number of principle components
        
        """
	self.debug = 0
        # Encode nominal features to conform with sklearn
        for i,tp in enumerate(dataSet.dtypes):
            if tp == 'object': 
                print 'Encoding feature \"' + dataSet.columns[i] + '\" ...'
                print 'Old dataset shape: ' + str(dataSet.shape)
                temp = pd.get_dummies(dataSet[dataSet.columns[i]],prefix=dataSet.columns[i])
                dataSet = pd.concat([dataSet,temp],axis=1).drop(dataSet.columns[i],axis=1)
                print 'New dataset shape: ' + str(dataSet.shape)
                #unique_vals, dataSet.ix[:,i]  = np.unique(dataSet.ix[:,i] , return_inverse=True)
                
	
        # Set the dependent variable (y) to the appropriate column
        y = dataSet.loc[:,dependentVar]
	names = list(dataSet)
	names.remove('backlog')
	
        # Transform that information to a format that scikit-learn understands
        # This may be redundant at times
        labels = preprocessing.LabelEncoder().fit_transform(y)
	
        # Remove the dependent variable from training sets
	X = dataSet.drop(dependentVar,1).values
	
        
	
        # Perform entropy-based feature selection 
	if doFeatureSelection:
            print 'Performing Feature Selection:'
            print 'Shape of dataset before feature selection: ' + str(X.shape)
            clf = DecisionTreeClassifier(criterion='entropy')
	    X = clf.fit(X, y).transform(X)

	    tree.export_graphviz(clf,out_file="tree.dot",feature_names = names)
	    
	    #if self.debug:print "Importance",clf.feature_importances_
	print "Selected Attributes are"
	features_all = features_sorted = selected = []
	features_all = [(name,round(imp,2))for name,imp in zip(names,clf.feature_importances_)]
	features_sorted = sorted(features_all, key=lambda x:x[1],reverse = True)
	for i in range(X.shape[1]):
		selected.append(features_sorted[i][0])
		print '{0} --> Importance: {1}'.format(features_sorted[i][0],features_sorted[i][1])   
	print 'Shape of dataset after feature selection: ' + str(X.shape) + '\n'
	    
        # Normalize values
        X = preprocessing.StandardScaler().fit(X).transform(X)
        
        # Save processed dataset, labels and student ids
        self.dataset = X
        self.labels = labels
        self.students = dataSet.index
	self.selected = selected

	
    def test(self,parameters, model):
	classifier = clfs[model]
	classifier.fit(self.dataset,self.labels)
	attr = [attributes[i]for i in self.selected]
	print self.selected
	final_parameters = [parameters[i]for i in attr]
	cnt = 0
	lst = []
	for i in attr:
		k = attribute[i]
		j = final_parameters[cnt]
		lst.append(k[j])
		cnt = cnt+1
	print lst
	result = classifier.predict(final_parameters)
	prob = classifier.predict_proba(final_parameters)
	prob = prob[0]
	course = course_id[parameters[attributes["course_id"]]]
	if result[0] == 1:
		return "You are predicted to have backlog in "+course
	else:
		return "Keep it up! You are predicted to clear "+course
	print "Your risk score is:",prob[1]*100,"\n"

    def runClassification(self, outputFormat='score', doSubsampling=False, subRate=1.0,
                            doSMOTE=False, pctSMOTE=100, nFolds=10, models=['LR'], topK=.1,):
        """ Main function to train and evaluate model

        Allows user to set the type of output and a few other parameters to running a K-fold
        cross validation experiment.        

        Parameters
        ----------
        outputFormat :  string
            The desired output format. Choices are: 'score', 'summary', 'matrix', 'roc', 'prc', 'topk' and 'risk'
        doSubsampling : bool
            Boolean value to determine whether to subsample the majority class
        subRate : float
            The ratio majority/minority to keep for training
        doSMOTE : bool
            Boolean value to determine whether or not to run SMOTE on the training set
        pctSMOTE : int
            The oversampling percentage to be used by SMOTE
        nFolds : int
            The number of folds to be assigned to the K-fold process    
        models : list
            A list of classifiers to evaluate given by the 2-3 letter codes above
	printFeatureImportance : bool (default False)
	    Boolean value to determine whether to print RF generated feature importances
        
        Returns
        --------
            Results are displayed inline for now
            
        """
        return_value = {}
        # Return a simple overall accuracy score
        if outputFormat=='score':
            # Iterate through each classifier to be evaluated
	    mean_scores = []
            for ix,clf in enumerate([clfs[x] for x in models]):
                kf = cross_validation.KFold(len(self.dataset), nFolds, shuffle=True)
                scores = cross_validation.cross_val_score(clf, self.dataset, self.labels, cv=kf)
		mean_scores.append(np.mean(scores))                
		if self.debug:print models[ix]+ ' Accuracy: %.2f' % np.mean(scores)
            model_selected = models[mean_scores.index(max(mean_scores))]
            clf = clfs[model_selected]

            # Return a summary table describing several metrics or a confusion matrix
            # Store the prediction results and their corresponding real labels for each fold

            y_prediction_results = []
            y_original_values = []

            # Generate indexes for the K-fold setup

            kf = cross_validation.StratifiedKFold(self.labels,
                    n_folds=nFolds)
            for (i, (train, test)) in enumerate(kf):
                if doSubsampling:

                    # Remove some random majority class instances to balance data

                    train = self.subsample(self.dataset,
                            self.labels, train, subRate)
            
                # Generate predictions for current hold-out sample in i-th fold

                fitted_clf = clf.fit(self.dataset[train],
                        self.labels[train])

        # self.feature_importances = getattr(fitted_clf, 'feature_importances_', None)

                y_pred = fitted_clf.predict(self.dataset[test])

                # Append results to previous ones

                y_prediction_results = \
                    np.concatenate((y_prediction_results, y_pred),
                        axis=0)

                # Store the corresponding original values for the predictions just generated

                y_original_values = \
                    np.concatenate((y_original_values,
                        self.labels[test]), axis=0)

            # Print result summary table based on k-fold
            # This is specific to our particular experiment and classes are hard coded
            # When oversampling is True, both results are displayed

            print '\t\t\t\t\t\t'+model_selected+ ' Summary Results'
            cm = classification_report(y_original_values, y_prediction_results,target_names=['Pass','Backlog'])
            print(str(cm)+'\n')
            print '----------------------------------------------------------\n'
            # Print the confusion matrix

            print '\t\t\t\t\t'+model_selected+ ' Confusion Matrix'
            print '\t\t\t\tPass\tBacklog'
            cm = confusion_matrix(y_original_values, y_prediction_results)
            print 'Pass\t\t\t%d\t\t%d'% (cm[0][0],cm[0][1])
            print 'Backlog\t%d\t\t%d'% (cm[1][0],cm[1][1])                       
            print '----------------------------------------------------------\n'
	    """
	    #-----------------------------------ROC-------------------------------------

	    kf = cross_validation.StratifiedKFold(self.labels,
                        n_folds=nFolds)
            mean_tpr = 0.0
            mean_fpr = np.linspace(0, 1, 100)

            for (i, (train, test)) in enumerate(kf):
		# Generate "probabilities" for the current hold out sample being predicted
		fitted_clf = clf.fit(self.dataset[train], self.labels[train])

		# self.feature_importances = getattr(fitted_clf, 'feature_importances_', None)
		probas_ = fitted_clf.predict_proba(self.dataset[test])

		# Compute ROC curve and area the curve
		(fpr, tpr, thresholds) = roc_curve(self.labels[test], probas_[:, 1])
		mean_tpr += np.interp(mean_fpr, fpr, tpr)

                # Plot ROC baseline

                pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6),
                        label='Baseline')

                # Compute true positive rates

                mean_tpr /= len(kf)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)

                # Plot results

                pl.plot(mean_fpr, mean_tpr, 'k-',
                        label='Mean ROC (area = %0.2f)' % mean_auc,
                        lw=2)

                # Plot results with oversampling

                pl.xlim([-0.05, 1.05])
                pl.ylim([-0.05, 1.05])
                pl.xlabel('False Positive Rate')
                pl.ylabel('True Positive Rate')
                pl.title(model_selected + ' ROC')
                pl.legend(loc='lower right')
                pl.show()
            """
	    print "Model Selected::",model_selected,"with accuracy:",max(mean_scores)
	    return_value["score"] = model_selected
        return return_value
