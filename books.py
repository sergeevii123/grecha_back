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
