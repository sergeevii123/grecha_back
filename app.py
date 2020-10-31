from typing import Optional, List

from fastapi import FastAPI

from books import Library
from kdfs import BuildKdfRec
from rec_utils import load_pickle
from recommender import RecommenderWrapper

LIBRARY_PATH = 'data/library.pkl'
RECOMMENDER_PATH = 'data/recommender.pkl'
LIBRARY_KDF_PATH = 'data/kdf_lib.pkl'
RECOMMENDER_KDF_PATH = 'data/w_kdf_rec.pkl'


def create_app():
    app = FastAPI()
    app.state.library = load_pickle(LIBRARY_PATH)
    app.state.reestr = load_pickle(LIBRARY_KDF_PATH)
    app.state.recommeder = load_pickle(RECOMMENDER_PATH)
    app.state.recommeder_kdf = load_pickle(RECOMMENDER_KDF_PATH)

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/books/{book_id}")
    def get_book(book_id: int):
        return app.state.library.get_book(book_id)

    @app.get("/items/{kdf_id}")
    def get_kdf(kdf_id: int):
        return app.state.reestr.get_kdf(kdf_id)

    @app.get("/get_all_kdf")
    def get_kdf():
        return [
                    {
                        'name':value.name, 'id':value.rec_id
                    }
            for key,value in app.state.reestr.kdfs.items()
        ]

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

    @app.get("/recs_kdf/{user_id}")
    def recommend_kdf(user_id: int, history_items: Optional[int] = 10, rec_items: Optional[int] = 10):
        recs = app.state.recommeder_kdf.recommend(user_id, rec_items)
        history = app.state.recommeder_kdf.history(user_id, history_items)
        kdfs = [
            app.state.reestr.get_kdf(kdf_id[0])
            for kdf_id in recs
        ]
        history = [
            app.state.reestr.get_kdf(kdf_id)
            for kdf_id in history
        ]
        return {
            'history': history,
            'kdfs': kdfs
        }

    @app.post("/build_kdf")
    def build_kdf(req: BuildKdfRec):
        recs = app.state.recommeder_kdf.get_pred_for_items(req.kdfs)
        kdfs = [
            app.state.reestr.get_kdf(kdf_id)
            for kdf_id in recs
        ]
        return {
            'kdfs': kdfs
        }

    return app



