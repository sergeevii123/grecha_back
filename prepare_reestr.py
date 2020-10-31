import pandas as pd

from app import LIBRARY_KDF_PATH
from kdfs import create_kdfs_info_dict, Reestr
from rec_utils import dump_pickle

if __name__ == '__main__':
    services = pd.read_csv('data/services.csv')
    kdfs = create_kdfs_info_dict(services)
    lib = Reestr(kdfs)
    dump_pickle(lib, LIBRARY_KDF_PATH)
