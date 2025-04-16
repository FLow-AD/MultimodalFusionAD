from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import shap

def predict_standalone(X_train, X_test, y_train, y_test):
  features = X_train.columns 
  BA_alone = []
  for var in features:
    Xtr = X_train[var].to_numpy()
    Xtst = X_test[var].to_numpy()
    model = LinearDiscriminantAnalysis(solver='svd') #it can be any ML models
    model.fit(Xtr, y_train)
    pr = model.predict(Xtst)
    BA_alone.append(balanced_accuracy_score(y_test, pr))

  weight_vector = 1 - np.max(BA_alone) + BA_alone
  weight_df = pd.DataFrame([features, BA_alone, weight_vector]).T
  return weight_df

def hybrid_weighted_concatenation(train_set, test_set, weight_df, modalities=['background', 'cognitive_scores', 'neuropsychological', 'csf', 'mri']):
    global multi_data # pre-determined dictionary where keys are modalities and values are the corresponding features

    weights = weight_df.to_dict('dict')['Weight']
    train_X = train_set.drop('y', axis=1)
    y_train = train_set['y']
    cols = list(train_X.columns)
    
    test_X = test_set.drop('y', axis=1)
    y_test = test_set['y']

    modal_weights = {}
    for modal in modalities:
        modal_cols = multi_data[modal]
        w_ = np.array([weights[var] for var in modal_cols])
        w_ = 1 - np.max(w_) + w_
        w_modal = dict(zip(modal_cols, w_))
        modal_weights.update(w_modal)

    # separate the categorical from numerical features
    categorical = [col for col in cols if train_X[col].dtype == 'category']
    numerical = list(set(cols) - set(categorical))

    num_tr_hwc = np.array([train_X[var].to_numpy()*weights[var] for var in numerical])
    num_tst_hwc = np.array([test_X[var].to_numpy()*weights[var] for var in numerical])

    cat_tr_hwc = train_X[categorical]
    cat_tst_hwc = test_X[categorical]
    
    num_tr_hwc = pd.DataFrame(num_tr_hwc.T, columns=numerical)
    X_train = pd.concat([cat_tr_hwc, num_tr_hwc], axis=1)
    num_tst_hwc = pd.DataFrame(num_tst_hwc.T, columns=numerical)
    X_test = pd.concat([cat_tst_hwc, num_tst_hwc], axis=1)

  return X_train, X_test, y_train, y_test

def fit_early(X, y):
  model = LinearDiscriminantAnalysis(solver='svd') #it can be any ML models
  return model.fit(X, y)

def interpret_model(X_test: np.ndarray, y_test: np.ndarray, model: BaseEstimator):
  class_names = ['Cognitive Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]
  model = fit_early(X_train, y_train)
  explainer = shap.LinearExplainer(model, X_test, output_names=class_names) # it shall fit the type of learners, e.g. Linear for Linear Discriminant Analysis
  shap_values = explainer.shap_values(X_test)
  return shap_values