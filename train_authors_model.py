import os
from collections import defaultdict

import implicit
import pandas as pd
import numpy as np
from implicit.nearest_neighbours import CosineRecommender
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from app import RECOMMENDER_PATH, USER_HISTORY_PATH, AUTHOR_RECOMMENDER_PATH, USER_GT_PATH
from rec_utils import dump_pickle
from recommender import RecommenderWrapper, AuthorTopItemsRecommender, SimilarAuthorRecommender
from train_utils import prepare_issues_dataset
from user_history import UserHistory


def get_authors_items(df: pd.DataFrame) -> dict:
    authors = defaultdict(list)
    df = df.groupby(['author', 'record_id'])[['reader_id']].count().reset_index()
    for tup in df.itertuples():
        authors[tup.author].append((tup.record_id, tup.reader_id))

    for author, records in authors.items():
        total_items = sum(record[1] for record in records)
        records = [(item, score / total_items) for item, score in records]
        authors[author] = records

    return authors


if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    issues_prepared = prepare_issues_dataset()

    user_lc = LabelEncoder()
    author_lc = LabelEncoder()
    item_lc = LabelEncoder()

    issues_prepared['reader_id'] = user_lc.fit_transform(issues_prepared['reader_id'])
    issues_prepared['author'] = author_lc.fit_transform(issues_prepared['author'])
    issues_prepared['record_id'] = item_lc.fit_transform(issues_prepared['record_id'])

    n_readers = issues_prepared.reader_id.max() + 1
    n_items = issues_prepared.author.max() + 1

    issues_train = issues_prepared[issues_prepared.issue_date < '2020-09-01']
    issues_test = issues_prepared[issues_prepared.issue_date >= '2020-09-01']

    issues_train = issues_train.groupby(['reader_id', 'author'])['record_id'].count().reset_index()
    issues_test = issues_test.groupby(['reader_id', 'author'])['record_id'].count().reset_index()

    train_matrix = csr_matrix(
        (issues_train['record_id'], (issues_train['reader_id'], issues_train['author'])),
        shape=(n_readers, n_items)
    ).astype('float64')

    test_matrix = csr_matrix(
        (issues_test['record_id'], (issues_test['reader_id'], issues_test['author'])),
        shape=(n_readers, n_items)
    ).astype('float64')

    model = CosineRecommender()
    model.fit(train_matrix.T)

    author_top_items = get_authors_items(issues_prepared)

    similar_author_recommender = SimilarAuthorRecommender(model, train_matrix)
    author_top_items_recommender = AuthorTopItemsRecommender(
        similar_author_recommender,
        author_top_items,
        None
    )

    wrapper = RecommenderWrapper(
        user_encoder=user_lc,
        item_encoder=item_lc,
        model=author_top_items_recommender
    )

    dump_pickle(wrapper, AUTHOR_RECOMMENDER_PATH)
