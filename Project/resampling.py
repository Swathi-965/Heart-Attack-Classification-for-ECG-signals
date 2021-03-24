import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

path = ""
mitbih_train = pd.read_csv(path+"mitbih_train.csv")
mitbih_train.columns = list(range(len(mitbih_train.columns)))
mitbih_test = pd.read_csv(path+"mitbih_test.csv")
mitbih_test.columns = list(range(len(mitbih_test.columns)))
mitbih_train[187] = mitbih_train[187].astype(int)
mitbih_test[187] = mitbih_test[187].astype(int)

mitbih_train0 = mitbih_train[mitbih_train[187]==0]
drop_indices = np.random.choice(mitbih_train0.index, len(mitbih_train0)-20000, replace=False)
mitbih_train = mitbih_train.drop(drop_indices)

trainx = mitbih_train.drop([187], axis=1)
trainy = mitbih_train[187]
testx = mitbih_test.drop([187], axis=1)
testy = mitbih_test[187]

sma = SMOTE('not majority', random_state=40)
trainx, trainy = sma.fit_resample(trainx, trainy)
trainx = np.expand_dims(trainx, 2)
testx = np.expand_dims(testx, 2)

print(trainx.shape)
print(trainy.shape)
print(testx.shape)
print(testy.shape)
trainn, dim, _ = trainx.shape
testn = testy.shape[0]
tags = 5

