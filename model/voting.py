from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .utils import evaluate_voting
from scipy import stats
import numpy as np

def fuse_decision_voting(train_set, test_set, model='soft-voting', modalities=['background', 'cognitive_scores', 'neuropsychological', 'csf', 'mri']):
    """note that train_set and test_set 
    have been arranged as dictionary 
    with modalities as the keys"""

    base = LinearDiscriminantAnalysis(solver='svd') #it can be any ML models
    prob_all, pred_all = None, None
    for modal in modalities:
        X_train = train_set[modal].drop('y', axis=1)
        X_test = test_set[modal].drop('y', axis=1)
        y_train = train_set[modal]['y']
        y_test = test_set[modal]['y']

        base.fit(X_train, y_train)

        y_pred = base.predict(X_test)
        prob = base.predict_proba(X_test)

        if prob_all is None:
            prob_all = prob
            pred_all = y_pred
        else:
            prob_all += prob
            pred_all = np.column_stack((pred_all, y_pred))

    if model == 'soft-voting':
        prob_all = softmax(prob_all)
        y_pred_all = np.argmax(prob_all, axis=1)
    elif model == 'hard-voting':
        prob_all = prob_all/len(modalities)
        y_pred_all = [stats.mode(pred_all[n], keepdims=True).mode for n in range(pred_all.shape[0])]
    
    metrics = evaluate_voting(y_test, y_pred_all, prob_all, average='weighted')

    output = {
        'predict_proba': prob_all,
        'predict': y_pred_all
    }
    return output, metrics

def softmax(z, epsilon=1e-5):
    return (np.exp(z).T / (np.sum(np.exp(z), axis=1) + epsilon)).T
