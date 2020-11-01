from pydantic import BaseModel
import pandas as pd
from typing import List, Union


class BuildKdfRec(BaseModel):
    kdfs: List[int]


class KDF(BaseModel):

    name: str
    kdf_id: int
    rec_id: int


class Reestr:

    def __init__(self, kdfs: dict, popular: list):
        self.kdfs = kdfs
        self.popular = popular

    def get_popular(self, topk):
        return self.popular[:topk]

    def get_kdf(self, rec_id: int):
        return self.kdfs.get(rec_id)


def create_kdfs_info_dict(services: pd.DataFrame, ) -> Union[dict, list]:
    kdfs = {}
    popularity = []
    errors = 0
    for row in services.iterrows():
        row = row[1]
        try:
            kdfs[row.id_enc_cluster] = KDF(name=row.clustered_name, kdf_id=row.id_clustered, rec_id=row.id_enc_cluster)
            if row.popularity is not None  and not pd.isna(row.popularity):
                popularity.append((row.popularity, kdfs[row.id_enc_cluster]))
        except:
            errors += 1
            continue

    print(f'total errors: {errors}')
    popularity = sorted(popularity,  key=lambda tup: tup[0], reverse=True)
    return kdfs, popularity
