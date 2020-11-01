from app import POPULAR_BOOKS_SAMPLER_PATH
from books import PopularBooksSampler
from rec_utils import dump_pickle
from train_utils import prepare_issues_dataset

if __name__ == '__main__':
    issues = prepare_issues_dataset()
    popularities = issues['record_id'].value_counts().to_frame().reset_index()
    popularities.columns = ['record_id', 'popularity']
    sampler = PopularBooksSampler(popularities.record_id, popularities.popularity)
    dump_pickle(sampler, POPULAR_BOOKS_SAMPLER_PATH)
