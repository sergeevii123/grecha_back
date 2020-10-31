from pydantic import BaseModel
import pandas as pd
from typing import List

class BuildKdfRec(BaseModel):
    kdfs: List[int]


class KDF(BaseModel):

    name: str
    kdf_id: int
    rec_id: int


class Reestr:

    def __init__(self, kdfs: dict):
        self.kdfs = kdfs

    def get_kdf(self, rec_id: int):
        return self.kdfs.get(rec_id)


def create_kdfs_info_dict(services: pd.DataFrame, ) -> dict:
    kdfs = {}
    errors = 0
    for row in services.iterrows():
        row = row[1]
        try:
            kdfs[row.id_enc_cluster] = KDF(name=row.clustered_name, kdf_id=row.id_clustered, rec_id=row.id_enc_cluster)
        except:
            errors += 1
            continue

    print(f'total errors: {errors}')

    return kdfs
