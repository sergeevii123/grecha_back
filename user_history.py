from scipy.sparse import csr_matrix


class UserHistory:

    def __init__(self, user_item_matrix: csr_matrix):
        self.user_item_matrix = user_item_matrix

    def get_user_history(self, user_id: int) -> set:
        return set(self.user_item_matrix[user_id].nonzero()[1])
