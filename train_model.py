import os

import implicit
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from app import RECOMMENDER_PATH, USER_HISTORY_PATH
from rec_utils import dump_pickle
from recommender import RecommenderWrapper, ImplicitRecommender
from train_utils import prepare_issues_dataset
from user_history import UserHistory

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

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

    model = implicit.als.AlternatingLeastSquares(factors=100, iterations=5)
    model.fit(train_matrix.T)

    wrapper = RecommenderWrapper(
        user_encoder=user_lc,
        item_encoder=item_lc,
        model=ImplicitRecommender(model, train_matrix)
    )
    user_history = UserHistory(train_matrix)

    dump_pickle(user_history, USER_HISTORY_PATH)
    dump_pickle(wrapper, RECOMMENDER_PATH)
