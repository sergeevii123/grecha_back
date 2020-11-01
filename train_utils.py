import pandas as pd


def prepare_issues_dataset():
    issues = pd.read_csv('data/issues.csv', parse_dates=['issue_date', 'return_date'])
    catalogue = pd.read_csv('data/catalogue.csv')
    examples = pd.read_csv('data/examples.csv')

    issues = issues.merge(examples[['record_id', 'barcode']]).merge(
        catalogue[['record_id', 'author']]
    )
    issues_dedup = issues.drop_duplicates(['reader_id', 'record_id']).reset_index()
    issues_prep = issues_dedup[['reader_id', 'record_id', 'issue_date', 'author']].copy().dropna()
    return issues_prep
