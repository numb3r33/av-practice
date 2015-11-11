import pandas as pd

mapping_dict = {1: 'Y', 0: 'N'}

def inverse_mapping(pred):
    return mapping_dict[pred]

def create_submissions(ids, preds, filename='baseline_submission.csv'):
    submission = pd.read_csv('./data/Sample_Submission_ZAuTl8O.csv')

    submission['Loan_ID'] = ids
    submission['Loan_Status'] = preds

    submission.to_csv('./submissions/' + filename, index=False)
