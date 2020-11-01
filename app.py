from typing import Optional

from fastapi import FastAPI

from rec_utils import load_pickle
from recommender import filter_genres

LIBRARY_PATH = 'data/library.pkl'
RECOMMENDER_PATH = 'data/recommender.pkl'
AUTHOR_RECOMMENDER_PATH = 'data/author_recommender.pkl'
USER_HISTORY_PATH = 'data/user_history.pkl'
USER_GT_PATH = 'data/user_gt.pkl'


def create_app():
    app = FastAPI()
    app.state.library = load_pickle(LIBRARY_PATH)
    app.state.recommeder = load_pickle(RECOMMENDER_PATH)
    app.state.author_recommender = load_pickle(AUTHOR_RECOMMENDER_PATH)
    app.state.user_gt = load_pickle(USER_GT_PATH)
    app.state.user_history = load_pickle(USER_HISTORY_PATH)
    app.state.author_recommender.user_history = app.state.user_history


    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/books/{book_id}")
    def get_book(book_id: int):
        return app.state.library.get_book(book_id)

    @app.get("/recs/{reader_id}")
    def recommend(reader_id: int, history_items: Optional[int] = 10, rec_items: Optional[int] = 10):
        recs = app.state.recommeder.recommend(reader_id, rec_items + 30)
        history = app.state.recommeder.history(reader_id, history_items)
        gt = app.state.user_gt.get_user_history(
            app.state.author_recommender.encoded_user_id(reader_id)
        )
        books = [
            app.state.library.get_book(book_id)
            for book_id in recs
        ]
        history = [
            app.state.library.get_book(book_id)
            for book_id in history
        ]
        gt = [
            app.state.library.get_book(book_id)
            for book_id in gt
        ]
        return {
            'history': history,
            'books': filter_genres(history, books)[:rec_items],
            'gt': gt,
        }

    @app.get("/recs_author/{reader_id}")
    def recommend_author(
            reader_id: int,
            history_items: Optional[int] = 10,
            rec_items: Optional[int] = 10
    ):
        recs = app.state.author_recommender.recommend(reader_id, rec_items + 30)
        history = app.state.author_recommender.history(reader_id, history_items)
        gt = app.state.user_gt.get_user_history(
            app.state.author_recommender.encoded_user_id(reader_id)
        )
        books = [
            app.state.library.get_book(book_id)
            for book_id in recs
        ]
        history = [
            app.state.library.get_book(book_id)
            for book_id in history
        ]
        gt = [
            app.state.library.get_book(book_id)
            for book_id in gt
        ]
        return {
            'history': history,
            'books': filter_genres(history, books)[:rec_items],
            'gt': gt,
        }

    return app



