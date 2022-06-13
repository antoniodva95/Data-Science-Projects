import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import pickle


class Encoding_Strategy_string(BaseEstimator,TransformerMixin):
    
    def __init__(self,column):
        self.column = column
        
        
    def fit(self,X,y=None):
        self.string_length = X[self.column].str.len().max()
        
        return self
        

    def transform(self,X):

        X_ = X.copy()

        for letter in range(self.string_length):

            X_["position"+ "_" + str(letter)] = X_[self.column].str.get(letter)
            X_["position"+ "_" + str(letter)] = X_["position"+ "_" + str(letter)].apply(lambda x: ord(x)-ord("A"))
            X_["len_unique_values"] = [len(set(s)) for s in X_[self.column]]
            
        X_.drop(columns=self.column,inplace=True)
        
        return X_

    

class Exploiting_found_interactions(BaseEstimator,TransformerMixin):
  
    def __init__(self):
      self.choices = [-1,0,1]  

    def fit(self,X,y=None):
        return self


    def transform(self,X):

      X_ = X.copy()


    

      X_["f_00_01"] = X_["f_00"] + X_["f_01"]

      conditions_1 = [(X_.f_21 + X_.f_02 < -5.3),(X_.f_21 + X_.f_02 >= -5.3) & (X_.f_21 + X_.f_02 <= 5.2),
                      (X_.f_21 + X_.f_02 > 5.2)]
      conditions_2 = [(X_.f_22 + X_.f_05 < -5.4),(X_.f_22 + X_.f_05 >= -5.3) & (X_.f_22 + X_.f_05 <= 5.1),
                      (X_.f_22 + X_.f_05 > 5.1)]
      conditions_3 = [(X_.f_00_01 + X_.f_26 < -5.3),(X_.f_00_01 + X_.f_26 >= -5.3) & 
                      (X_.f_00_01 + X_.f_26 <= 5.0),
                      (X_.f_00_01 + X_.f_26 > 5.0)]

      X_["f_02_21"] = np.select(conditions_1,self.choices)

      X_["f_05_22"] = np.select(conditions_2,self.choices)

      X_["f_00_01_26"] = np.select(conditions_3,self.choices)



      return X_

#Loading train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Preparing X and y for training pipeline
X = train.drop(columns='target')
y = train['target']

X = X.drop(columns='id')


# At the end of the day, best model seems to be LGBM with 5000 estimators. Let's fit it. 

# Production model

Production_pipe = Pipeline(steps= [
    ("es", Encoding_Strategy_string(column="f_27")),
    ("interaction", Exploiting_found_interactions()),
    ("norm", StandardScaler()),
    ("clf_5", LGBMClassifier(n_estimators=5000,min_child_samples=80))
],verbose=True)

Production_Model = Production_pipe.fit(X,y)


# Let's make predictions
ids = test['id']
X_test = test.drop(columns='id')


predictions = Production_Model.predict(X_test)
predictions_series = pd.Series(predictions)
predictions_dataframe = pd.concat([ids,predictions_series],axis=1)
predictions_dataframe = predictions_dataframe.rename(columns={0:"predictions"})


filepath = 'Choose your path'


with open(filepath + 'finalized_model.sav','wb') as f:
  pickle.dump(Production_Model,f)


predictions_dataframe.to_csv('wherever the path/predictions_test.csv')

