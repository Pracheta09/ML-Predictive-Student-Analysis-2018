import classification
import pandas as pd

# Load simulated data set for experiment
df = pd.read_csv('data.csv',index_col=0)

# Create a model object using the loaded data
pred = classification.Model(df,'backlog')

# Run classification using 10-fold cross validation
# Classifier used: Logistic Regression (LR)
# Output format:List of risk scores for the top 5% of students at highest risk
pred.runClassification(outputFormat='score', models=['LR','RF','ET','AB','SVM','GB','NB','DT'], nFolds=10)

