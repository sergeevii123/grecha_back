import pandas as pd

from app import LIBRARY_PATH
from books import create_books_info_dict, Library
from rec_utils import dump_pickle

if __name__ == '__main__':
    catalogue = pd.read_csv('data/catalogue.csv')
    books = create_books_info_dict(catalogue)
    lib = Library(books)
    dump_pickle(lib, LIBRARY_PATH)
