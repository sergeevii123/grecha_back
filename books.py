from random import choices

from pydantic import BaseModel
import pandas as pd


class Book(BaseModel):

    record_id: int
    author: str
    title: str
    year: str
    genres: str


class Library:

    def __init__(self, books: dict):
        self.books = books

    def get_book(self, book_id: int):
        return self.books.get(book_id)


class PopularBooksSampler:

    def __init__(self, book_ids: list, popularities: list, power: float = 1):
        self.book_ids = book_ids
        self.popularities = [x ** power for x in popularities]

    def sample_books(self, n_books: int = 30):
        return choices(self.book_ids, self.popularities, k=n_books)


def create_books_info_dict(catalogue: pd.DataFrame) -> dict:
    books = {}
    errors = 0
    for tup in catalogue.itertuples():
        try:
            books[tup.record_id] = Book(**tup._asdict())
        except:
            errors += 1
            continue

    print(f'total errors: {errors}')

    return books
