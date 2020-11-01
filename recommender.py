from abc import ABC, abstractmethod
from operator import itemgetter
from typing import List, Optional

from implicit.nearest_neighbours import ItemItemRecommender
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares

from books import Book
from user_history import UserHistory


class BaseRecommender(ABC):

    @abstractmethod
    def recommend(self, user_id: List[int], num_recs: Optional[int] = 10) -> list:
        pass

    def history(self, user_id: List[int], num_items: Optional[int] = 10) -> list:
        pass


class ImplicitRecommender(BaseRecommender):

    def __init__(self, als_model: AlternatingLeastSquares, user_items: csr_matrix):
        self.als_model = als_model
        self.user_items = user_items

    def recommend(self, user_id: List[int], num_recs: Optional[int] = 10) -> list:
        return self.als_model.recommend(user_id, self.user_items, N=num_recs)

    def history(self, user_id: List[int], num_items: int = 10) -> list:
        return self.user_items[user_id].nonzero()[1][-num_items:]


class PopularRecommender(BaseRecommender):

    def __init__(self, top_items: List[int]):
        self.top_items = top_items

    def recommend(self, user_id: List[int], num_recs: int = 10) -> list:
        return self.top_items[:num_recs]


class SimilarAuthorRecommender(BaseRecommender):

    def __init__(
            self,
            similarity_model: ItemItemRecommender,
            user_authors: csr_matrix
    ):
        self.similarity_model = similarity_model
        self.user_authors = user_authors

    def recommend(self, user_id: List[int], num_recs: Optional[int] = 10) -> list:
        return self.similarity_model.recommend(
            user_id,
            self.user_authors,
            N=num_recs,
            filter_already_liked_items=False
        )

    def history(self, user_id: List[int], num_items: int = 10) -> list:
        return self.user_authors[user_id].nonzero()[1][-num_items:]


class AuthorTopItemsRecommender:

    def __init__(
            self,
            model: SimilarAuthorRecommender,
            author_top_items: dict,
            user_history: UserHistory,
    ):
        self.model = model
        self.author_top_items = author_top_items
        self.user_history = user_history

    def recommend(self, user_id: int, num_items: int = 10):
        recs = self.model.recommend(user_id)
        candidates = []
        for author_id, score in recs:
            author_items = self.author_top_items.get(author_id, [])
            rec = sorted(
                [
                    (item_id, pop)
                    for item_id, pop in author_items
                    if item_id not in self.user_history.get_user_history(user_id)
                ],
                key=itemgetter(1),
                reverse=True,
            )[0]

            candidates.append(rec)

        return candidates[:num_items]

    def history(self, user_id: List[int], num_items: int = 10) -> list:
        return list(self.user_history.get_user_history(user_id))[-num_items:]


class RecommenderWrapper:

    def __init__(
            self,
            user_encoder: LabelEncoder,
            item_encoder: LabelEncoder,
            model: BaseRecommender,
    ):
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.model = model

    def encoded_user_id(self, user_id):
        return self.user_encoder.transform([user_id])[0]

    def recommend(self, user_id: int, num_items: int = 10) -> list:
        encoded_user_id = self.encoded_user_id(user_id)
        recs = self.model.recommend(encoded_user_id, num_items)
        rec_ids = [rec[0] for rec in recs]
        return self.item_encoder.inverse_transform(rec_ids)

    def history(self, user_id: int, num_items: int = 10) -> list:
        encoded_user_id = self.encoded_user_id(user_id)
        item_ids = self.model.history(encoded_user_id, num_items)
        return self.item_encoder.inverse_transform(item_ids)


def filter_genres(history: List[Book], recs: List[Book]):
    history_genres = {
        book.genres
        for book in history
    }
    return [
        book for book in recs if book.genres in history_genres
    ]
