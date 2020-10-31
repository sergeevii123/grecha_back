from typing import Optional

from fastapi import FastAPI

from rec_utils import load_pickle

LIBRARY_PATH = 'data/library.pkl'
RECOMMENDER_PATH = 'data/recommender.pkl'
USER_HISTORY_PATH = 'data/user_history.pkl'


def create_app():
    app = FastAPI()
    app.state.library = load_pickle(LIBRARY_PATH)
    app.state.recommeder = load_pickle(RECOMMENDER_PATH)

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/books/{book_id}")
    def get_book(book_id: int):
        return app.state.library.get_book(book_id)

    @app.get("/recs/{reader_id}")
    def recommend(reader_id: int, history_items: Optional[int] = 10, rec_items: Optional[int] = 10):
        recs = app.state.recommeder.recommend(reader_id, rec_items)
        history = app.state.recommeder.history(reader_id, history_items)
        books = [
            app.state.library.get_book(book_id)
            for book_id in recs
        ]
        history = [
            app.state.library.get_book(book_id)
            for book_id in history
        ]
        return {
            'history': history,
            'books': books
        }

    return app



