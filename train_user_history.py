import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from app import USER_HISTORY_PATH, USER_GT_PATH
from rec_utils import dump_pickle
from train_utils import prepare_issues_dataset
from user_history import UserHistory

if __name__ == '__main__':

    issues_prepared = prepare_issues_dataset()

    user_lc = LabelEncoder()
    item_lc = LabelEncoder()

    issues_prepared['reader_id'] = user_lc.fit_transform(issues_prepared['reader_id'])
    issues_prepared['record_id'] = item_lc.fit_transform(issues_prepared['record_id'])

    n_readers = issues_prepared.reader_id.max() + 1
    n_items = issues_prepared.record_id.max() + 1

    issues_train = issues_prepared[issues_prepared.issue_date < '2020-09-01']
    issues_test = issues_prepared[issues_prepared.issue_date >= '2020-09-01']

    train_matrix = csr_matrix(
        (np.ones(len(issues_train)), (issues_train['reader_id'], issues_train['record_id'])),
        shape=(n_readers, n_items)
    )

    test_matrix = csr_matrix(
        (np.ones(len(issues_test)), (issues_test['reader_id'], issues_test['record_id'])),
        shape=(n_readers, n_items)
    )

    user_history = UserHistory(train_matrix)
    user_gt = UserHistory(test_matrix)

    dump_pickle(user_history, USER_HISTORY_PATH)
    dump_pickle(user_gt, USER_GT_PATH)
