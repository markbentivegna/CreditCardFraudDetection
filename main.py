import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import xgboost as xgb

selected_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
credit_df = pd.read_csv('creditcard.csv')

fraud_split_index = 250
valid_split_index = 150000

for col in credit_df.select_dtypes(include=['object']).columns:
    credit_df[col] = credit_df[col].fillna(value=0)
    credit_df[col] = pd.Categorical(credit_df[col], categories=credit_df[col].unique()).codes

fraud_indexes = credit_df[credit_df['Class'] == 1].index[:fraud_split_index]
valid_indexes = credit_df[credit_df['Class'] == 0].index[:valid_split_index]

fraud_test_indexes = credit_df[credit_df['Class'] == 1].index[fraud_split_index:]
valid_test_indexes = credit_df[credit_df['Class'] == 0].index[valid_split_index:]

min_max_scaler = preprocessing.MinMaxScaler()
X = pd.DataFrame(min_max_scaler.fit_transform(credit_df), columns=credit_df.columns, index=credit_df.index).drop('Class', axis=1)[selected_features]
y = credit_df['Class']

# clf = BaggingClassifier(base_estimator=AdaBoostClassifier(), n_estimators=100, random_state=0)
clf = xgb.XGBClassifier(random_state=0)
for i in range(10):
    random_fraud_index = np.random.choice(fraud_indexes, 50)
    random_valid_index = np.random.choice(valid_indexes, 100)
    train_subset = np.concatenate((random_fraud_index.astype(int), random_valid_index.astype(int)), axis=0)
    clf.fit(X.iloc[train_subset.astype(int)], y.iloc[train_subset.astype(int)])
    
# random_valid_test_index = np.random.choice(valid_test_indexes, 30000)
test_subset = np.concatenate((fraud_test_indexes, valid_test_indexes), axis=0)
X_test = X.iloc[test_subset.astype(int)]
pred = clf.predict(X_test)

# print(accuracy_score(pred, y.iloc[test_subset]))
print(confusion_matrix(pred,y.iloc[test_subset]))
# print(recall_score(pred,y.iloc[test_subset]))
# print(precision_score(pred,y.iloc[test_subset]))
# print(f1_score(pred,y.iloc[test_subset]))
print(classification_report(pred, y.iloc[test_subset]))
# 0.902027027027027
# [[272  30]
#  [ 28 262]]
# 0.903448275862069
# 0.8972602739726028
