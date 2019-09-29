import classification
import pandas as pd


def test(self,parameters):
	# Load simulated data set for experiment
	df = pd.read_csv('../model/data.csv',index_col=0)
	# Create a model object using the loaded data
	pred = classification.Model(df,'backlog')
	# Run classification using 10-fold cross validation
	# Classifier used: Logistic Regression (LR)
	# Output format:List of risk scores for the top 5% of students at highest risk
	result = pred.runClassification(outputFormat='score', models=['LR','RF','ET','AB','SVM','GB','NB','DT'], nFolds=10)

	#Except first and backlog feature-- rest all
	return pred.test(parameters, result["score"]) #row5 data
	#pred.test([2,2,0,1,0,3,2,0,0,2,0,1,0,1,0,1,1,1,2,1,0,23,2,-1],result["score"]) #row2 data


