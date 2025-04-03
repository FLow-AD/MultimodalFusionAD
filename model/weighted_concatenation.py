from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from logitboost import LogitBoost
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np

def predict_standalone(X_train, X_test, y_train, y_test, random_seed=123):
  features = X_train.columns 
  BA_alone = []
  for var in features:
    Xtr = X_train[var].to_numpy()
    Xtst = X_test[var].to_numpy()
    model = LinearDiscriminantAnalysis(solver='svd', store_covariance=True) #it can be any ML models
    model.fit(Xtr, y_train)
    pr = model.predict(Xtst)
    BA_alone.append(balanced_accuracy_score(y_test, pr))

  weight_vector = 1 - np.max(BA_alone) + BA_alone
  weight_df = pd.DataFrame([features, BA_alone, weight_vector]).T
  return weight_df

def weighted_concatenation(train_set, test_set, weight_df, classifier='SVM - Linear'):
    weights = weight_df.to_dict('dict')['Weight']
    train_X = train_set.drop('y', axis=1)
    train_y = train_set['y']
    cols = list(train_X.columns)
    
    test_X = test_set.drop('y', axis=1)
    test_y = test_set['y']

    # separate the categorical from numerical features
    categorical = [col for col in cols if train_X[col].dtype == 'category']
    numerical = list(set(cols) - set(categorical))

    num_tr_swc = np.array([train_X[var].to_numpy()*weights[var] for var in numerical])
    num_tst_swc = np.array([test_X[var].to_numpy()*weights[var] for var in numerical])

    cat_tr_swc = train_X[categorical]
    cat_tst_swc = test_X[categorical]
    
    num_tr_swc = pd.DataFrame(num_tr_swc.T, columns=numerical)
    tr_swc = pd.concat([cat_tr_swc, num_tr_swc], axis=1)
    num_tst_swc = pd.DataFrame(num_tst_swc.T, columns=numerical)
    tst_swc = pd.concat([cat_tst_swc, num_tst_swc], axis=1)

  return tr_swc, tst_swc, train_y, test_y

def fit_early(X, y, random_seed=123):
  model = LinearDiscriminantAnalysis(solver='svd', store_covariance=True) #it can be any ML models
  return model.fit(X, y)

def interpret_model(X_test: np.ndarray, y_test: np.ndarray, model: BaseEstimator, random_seed=123):
  class_names = ['Cognitive Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]
  explainer = shap.LinearExplainer(model, X_test, output_names=class_names) # it shall fit the type of models, e.g. Linear for Linear Discriminant Analysis
  shap_values = explainer.shap_values(X_test)
  return shap_values
