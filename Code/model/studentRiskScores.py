import classification
import pandas as pd
import csv
import pickle

#To calculate risk score for each row
'''
df = pd.read_csv('../model/data.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

risk = []
def crawl(parameters):
	risk.append(pred.test(parameters, result["score"])) #row5 data

df = pd.read_csv('../model/data.csv',usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
pred = classification.Model(df,'backlog')
result = pred.runClassification(outputFormat='score', models=['LR','RF','ET','AB','SVM','GB','NB','DT'], nFolds=10)
df.apply(crawl, axis=1)
df['risk_score'] = risk
df.to_csv('new_data.csv')

'''

#Previous version
'''
# Load simulated data set for experiment
df = pd.read_csv('../model/data.csv',index_col=0)
# Create a model object using the loaded data
pred = classification.Model(df,'backlog')

#Except first and backlog feature-- rest all
print pred.test(parameters, result["score"]) #row5 data
#pred.test([2,2,0,1,0,3,2,0,0,2,0,1,0,1,0,1,1,1,2,1,0,23,2,-1],result["score"]) #row2 data
'''
#[2,2,0,1,0,3,2,0,0,2,0,1,0,1,0,1,1,1,2,1,0,20,1,6]

def test_predict(parameters):

	filename = '../model/finalized_model.sav'
	clf = pickle.load(open(filename, 'rb'))
	result = clf.predict(parameters)
	prob = clf.predict_proba(parameters)
	prob = prob[0]
	print "Your risk score is:",prob[1]*100,"\n"
	if result[0] == 1:
		return "You will fail the subject with risk: ",prob[1]*100
	else:
		return "You will clear the subject with risk: ",prob[1]*100

