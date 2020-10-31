from app import RECOMMENDER_KDF_PATH
from recommender import KDFRecommender
import pickle
import scipy
from rec_utils import dump_pickle


if __name__ == '__main__':
    with open('data/kdf_rec.pkl', 'rb') as f:
        model = pickle.load(f)
    sparse_matrix = scipy.sparse.load_npz('data/sparse_kdf.npz')
    w_rec = KDFRecommender(model, sparse_matrix)
    dump_pickle(w_rec, RECOMMENDER_KDF_PATH)
