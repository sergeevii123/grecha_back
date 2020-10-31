from abc import ABC, abstractmethod
from typing import List, Optional

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender


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
