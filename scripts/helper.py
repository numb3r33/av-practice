import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split

mapping_dict = {1: 'Y', 0: 'N'}

def inverse_mapping(pred):
    return mapping_dict[pred]

def create_submissions(ids, preds, filename='baseline_submission.csv'):
    submission = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')

    submission['Loan_ID'] = ids
    submission['Loan_Status'] = preds

    submission.to_csv('./submissions/' + filename, index=False)

def binary_from_prob(preds, threshold=0.5):
    return np.array(['Y' if pred > threshold else 'N' for pred in preds])

def get_mask(dataset, train_size=0.7):
    """
    Returns the boolean mask which can be used to split the dataset
    """

    loantrain, loantest = train_test_split(xrange(dataset.shape[0]), train_size=train_size)
    loanmask = np.ones(dataset.shape[0], dtype='int')
    loanmask[loantrain] = 1
    loanmask[loantest] = 0
    loanmask = (loanmask==1)

    return loanmask

def split_dataset(X, y):
    """
    Splits the training dataset into X_train and X_val
    """

    loanmask = get_mask(X)

    X_train = X[loanmask]
    y_train = y[loanmask]

    X_val = X[~loanmask]
    y_val = y[~loanmask]

    return (X_train, X_val, y_train, y_val)
