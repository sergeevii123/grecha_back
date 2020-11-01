from abc import ABC, abstractmethod
from collections import defaultdict
from operator import itemgetter
from typing import List, Optional

from implicit.nearest_neighbours import ItemItemRecommender
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender

from books import Book
from user_history import UserHistory


class BaseRecommender(ABC):

    @abstractmethod
    def recommend(self, user_id: int, num_recs: Optional[int] = 10) -> list:
        pass

    def history(self, user_id: int, num_items: Optional[int] = 10) -> list:
        pass


class BaseItemRecommender(ABC):

    @abstractmethod
    def recommend(self, history: List[int], num_recs: Optional[int] = 10) -> list:
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


class ALSSimilarityRecommender(BaseItemRecommender):

    def __init__(self, als_model: AlternatingLeastSquares, item_encoder: LabelEncoder):
        self.als_model = als_model
        self.item_encoder = item_encoder

    def recommend(self, history: List[int], num_recs: Optional[int] = 10) -> list:
        all_items = defaultdict(float)
        history_encoded = self.item_encoder.transform(history)
        history_set = set(history_encoded)
        for book_id in history_encoded:
            similar_items = self.als_model.similar_items(book_id)
            for item, score in similar_items:
                if item not in history_set:
                    all_items[item] += score

        recs = list(all_items.items())
        recs_sorted = sorted(recs, key=itemgetter(1), reverse=True)[:num_recs]
        return self.item_encoder.inverse_transform([rec[0] for rec in recs_sorted])


class KDFRecommender(BaseRecommender):

    def __init__(self, model: CosineRecommender, user_items: csr_matrix):
        self.model = model
        self.user_items = user_items

    def recommend(self, user_id: List[int], num_recs: Optional[int] = 10) -> list:
        return self.model.recommend(user_id, self.user_items, N=num_recs)

    def history(self, user_id: List[int], num_items: int = 10) -> list:
        return self.user_items[user_id].nonzero()[1][-num_items:]

    def get_pred_for_items(self, items, topk=10):
        out = []
        items_set = set()
        for item in items:
            items_set.add(item)

            for sim_i in self.model.similar_items(item, (topk // len(items)) + 1):
                if sim_i[0] not in items_set:
                    out.append(sim_i[0])

        return out[:topk]


def filter_genres(history: List[Book], recs: List[Book]):
    history_genres = {
        book.genres
        for book in history
    }
    return [
        book for book in recs if book.genres in history_genres
    ]
