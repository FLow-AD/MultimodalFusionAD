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

class FeatureFusion:
    """Handles multimodal feature fusion with multiple concatenation options"""
    
    def __init__(self, modalities=None, multi_data=None):
        self.modalities = modalities or ['background', 'cognitive_scores', 
                                       'neuropsychological', 'csf', 'mri']
        self.multi_data = multi_data or {}
        self.modal_weights = None
        
    def process_data(self, train_set, test_set, weight_df=None, method='simple'):
        """
        Process data using specified concatenation method
        
        Parameters:
        - train_set: Training data (DataFrame)
        - test_set: Test data (DataFrame)
        - weight_df: DataFrame with feature weights (required for weighted methods)
        - method: Concatenation method ('simple', 'weighted', 'hybrid')
        
        Returns:
        - Processed training and test data with labels
        """
        if method == 'simple':
            return self._simple_concatenation(train_set, test_set)
        elif method == 'weighted':
            if weight_df is None:
                raise ValueError("Weight dataframe required for weighted concatenation")
            return self._weighted_concatenation(train_set, test_set, weight_df)
        elif method == 'hybrid':
            if weight_df is None:
                raise ValueError("Weight dataframe required for hybrid concatenation")
            return self._hybrid_weighted_concatenation(train_set, test_set, weight_df)
        else:
            raise ValueError(f"Unknown concatenation method: {method}")

    def _simple_concatenation(self, train_set, test_set):
        """Simple concatenation without any weighting"""
        X_train = train_set.drop('y', axis=1)
        y_train = train_set['y']
        X_test = test_set.drop('y', axis=1)
        y_test = test_set['y']
        
        return X_train, X_test, y_train, y_test
    
    def _weighted_concatenation(self, train_set, test_set, weight_df):
        """Weighted concatenation using feature weights"""
        weights = weight_df.set_index('Feature')['Weight'].to_dict()
        
        train_X = train_set.drop('y', axis=1)
        y_train = train_set['y']
        cols = list(train_X.columns)
        
        test_X = test_set.drop('y', axis=1)
        y_test = test_set['y']

        # Separate categorical from numerical features
        categorical = [col for col in cols if train_X[col].dtype == 'category']
        numerical = list(set(cols) - set(categorical))

        # Apply weights to numerical features
        num_tr = np.array([train_X[var].to_numpy() * weights.get(var, 1) for var in numerical])
        num_tst = np.array([test_X[var].to_numpy() * weights.get(var, 1) for var in numerical])

        # Handle categorical features
        cat_tr = train_X[categorical]
        cat_tst = test_X[categorical]
        
        # Combine features
        num_tr = pd.DataFrame(num_tr.T, columns=numerical)
        X_train = pd.concat([cat_tr, num_tr], axis=1)
        num_tst = pd.DataFrame(num_tst.T, columns=numerical)
        X_test = pd.concat([cat_tst, num_tst], axis=1)

        return X_train, X_test, y_train, y_test
    
    def _hybrid_weighted_concatenation(self, train_set, test_set, weight_df):
        """Hybrid weighted concatenation with modality-specific weighting"""
        weights = weight_df.set_index('Feature')['Weight'].to_dict()
        
        train_X = train_set.drop('y', axis=1)
        y_train = train_set['y']
        cols = list(train_X.columns)
        
        test_X = test_set.drop('y', axis=1)
        y_test = test_set['y']

        # Calculate modality-specific weights
        self.modal_weights = {}
        for modal in self.modalities:
            modal_cols = self.multi_data.get(modal, [])
            w_ = np.array([weights[var] for var in modal_cols if var in weights])
            if len(w_) > 0:
                w_ = 1 - np.max(w_) + w_  # Normalize within modality
                w_modal = dict(zip(modal_cols, w_))
                self.modal_weights.update(w_modal)

        # Separate categorical from numerical features
        categorical = [col for col in cols if train_X[col].dtype == 'category']
        numerical = list(set(cols) - set(categorical))

        # Apply hybrid weights (modality weights where available, else feature weights)
        num_tr = np.array([
            train_X[var].to_numpy() * self.modal_weights.get(var, weights.get(var, 1))
            for var in numerical
        ])
        num_tst = np.array([
            test_X[var].to_numpy() * self.modal_weights.get(var, weights.get(var, 1))
            for var in numerical
        ])

        # Handle categorical features
        cat_tr = train_X[categorical]
        cat_tst = test_X[categorical]
        
        # Combine features
        num_tr = pd.DataFrame(num_tr.T, columns=numerical)
        X_train = pd.concat([cat_tr, num_tr], axis=1)
        num_tst = pd.DataFrame(num_tst.T, columns=numerical)
        X_test = pd.concat([cat_tst, num_tst], axis=1)

        return X_train, X_test, y_train, y_test

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

def fit_early(X, y, random_seed=123):
  model = LinearDiscriminantAnalysis(solver='svd', store_covariance=True) #it can be any ML models
  return model.fit(X, y)

def interpret_model(X_test: np.ndarray, y_test: np.ndarray, model: BaseEstimator, random_seed=123):
  class_names = ['Cognitive Normal', 'Mild Cognitive Impairment', "Alzheimer's Disease"]
  explainer = shap.LinearExplainer(model, X_test, output_names=class_names) # it shall fit the type of models, e.g. Linear for Linear Discriminant Analysis
  shap_values = explainer.shap_values(X_test)
  return shap_values
