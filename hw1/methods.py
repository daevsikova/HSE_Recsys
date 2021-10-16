import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


class Test:
    def __init__(self, items_info, user_emb, item_emb, users, items, user_bias=None, item_bias=None):
        self.items_info = items_info
        self.item_emb = item_emb
        self.user_emb = user_emb
        self.users = users
        self.items = items
        self.user_bias = user_bias
        self.item_bias = item_bias

    @staticmethod
    def cosine(x, y):
        return x @ y.T / (np.linalg.norm(x) * np.linalg.norm(y, axis=1) + 1e-8)

    def get_similars(self, item_id, k=10):
        i = np.where(self.items == item_id)[0][0]
        cur_item = self.item_emb[i]
        similars = self.cosine(cur_item, self.item_emb)

        # get top(k+1) items with max cosine similarity,
        # then remove query-item from list to leave only k-elements
        similars_ids = np.argpartition(similars, -k - 1)[-k - 1 :]

        nearest = self.items[similars_ids]

        # remove query item from list of nearest items
        result = self.items_info[self.items_info["movie_id"].isin(nearest) & ~(self.items_info["movie_id"] == item_id)]
        item = self.items_info[self.items_info["movie_id"] == item_id].name.values[0]
        return (item, result)

    def get_recommendations(self, user_id, k=10):
        i = np.where(self.users == user_id)[0][0]
        cur_user = self.user_emb[i]

        recs = cur_user @ self.item_emb.T
        if self.user_bias is not None:
            recs += self.user_bias[i]
        if self.item_bias is not None:
            recs += self.item_bias

        # get top-k max dot products
        recs_ids = np.argpartition(recs, -k)[-k:]
        best_K = self.items[recs_ids]

        result = self.items_info[self.items_info["movie_id"].isin(best_K)]
        return result


class SVD:
    def __init__(self, latent_size=32, lambd=1e-5, learning_rate=1e-2, epochs=10):
        self.lambd = lambd
        self.latent_size = latent_size
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, ratings):
        self.ratings = ratings
        self.n = self.ratings.shape[0]
        self.indexes = self.ratings.index.values
        self.users, self.items = np.unique(self.ratings["user_id"].values), np.unique(self.ratings["movie_id"].values)
        self.num_users, self.num_items = len(self.users), len(self.items)

        self.U = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_users, self.latent_size))
        self.V = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_items, self.latent_size))
        self.U_bias = np.zeros((self.num_users))
        self.V_bias = np.zeros((self.num_items))

        self.total_rmse = 0
        for epoch in range(self.epochs):
            indexes_perm = np.random.permutation(self.indexes)
            mse = 0

            for idx in tqdm(indexes_perm):
                user_item = self.ratings.loc[idx]
                i = np.where(self.users == user_item.user_id)[0][0]
                j = np.where(self.items == user_item.movie_id)[0][0]
                x = user_item.rating

                error = self.U[i] @ self.V[j] + self.U_bias[i] + self.V_bias[j] - x

                self.U[i] = self.U[i] - self.lr * (error * self.V[j] + self.lambd * self.U[i])
                self.U_bias[i] = self.U_bias[i] - self.lr * (error + self.lambd * self.U_bias[i])

                self.V[j] = self.V[j] - self.lr * (error * self.U[i] + self.lambd * self.V[j])
                self.V_bias[j] = self.V_bias[j] - self.lr * (error + self.lambd * self.V_bias[j])

                mse += error ** 2

            mse /= self.n
            rmse = mse ** 0.5
            print(f"epoch {epoch + 1}: RMSE = {rmse:.4f}")

            self.total_rmse += rmse
        self.total_rmse = self.total_rmse / self.epochs


class ALS:
    def __init__(self, latent_size=64, lambd=1e-5, epochs=20):
        self.lambd = lambd
        self.latent_size = latent_size
        self.epochs = epochs

    def _get_csr_matrix(self, dense_matrix):
        implicit_ratings = dense_matrix.loc[(dense_matrix["rating"] >= 4)]

        users = implicit_ratings["unique_user"]
        movies = implicit_ratings["unique_movie"]
        user_item = sp.coo_matrix((np.ones_like(users), (users, movies)))
        user_item_csr = user_item.tocsr()
        return user_item_csr

    def fit(self, ratings):
        self.ratings = ratings
        self.ratings["unique_movie"] = self.ratings.groupby("movie_id").ngroup()
        self.ratings["unique_user"] = self.ratings.groupby("user_id").ngroup()
        self.n = self.ratings.shape[0]
        self.users, self.items = np.unique(self.ratings["user_id"].values), np.unique(self.ratings["movie_id"].values)
        self.num_users, self.num_items = len(self.users), len(self.items)

        self.U = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_users, self.latent_size))
        self.V = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_items, self.latent_size))

        self.sparse_ratings = self._get_csr_matrix(self.ratings)
        rows, cols = self.sparse_ratings.nonzero()

        self.total_rmse = 0
        for epoch in range(self.epochs):

            self.U = (
                np.linalg.inv(self.V.T @ self.V + self.lambd * np.eye(self.latent_size))
                @ self.V.T
                @ self.sparse_ratings.T
            ).T

            self.V = (
                np.linalg.inv(self.U.T @ self.U + self.lambd * np.eye(self.latent_size))
                @ self.U.T
                @ self.sparse_ratings
            ).T

            rmse = np.power((self.U @ self.V.T)[rows, cols] - self.sparse_ratings[rows, cols], 2).mean() ** 0.5
            print(f"epoch {epoch + 1}: RMSE = {rmse:.4f}")

            self.total_rmse += rmse

        self.total_rmse = self.total_rmse / self.epochs


class BPR:
    def __init__(self, latent_size=64, lambd=1e-5, learning_rate=1e-3, epochs=10):
        self.lambd = lambd
        self.latent_size = latent_size
        self.lr = learning_rate
        self.epochs = epochs

    def _get_ui_pairs(self):
        data = []
        for user in range(self.num_users):
            _, i_range = self.sparse_ratings[user].nonzero()
            data.extend([[user, i] for i in i_range])
        return np.array(data)

    def _get_csr_matrix(self, dense_matrix):
        implicit_ratings = dense_matrix.loc[(dense_matrix["rating"] >= 4)]

        users = implicit_ratings["unique_user"]
        movies = implicit_ratings["unique_movie"]
        user_item = sp.coo_matrix((np.ones_like(users), (users, movies)))
        user_item_csr = user_item.tocsr()
        return user_item_csr

    @staticmethod
    def sigmoid(x):
        return np.exp(-x + 1e-8) / (1 + np.exp(-x + 1e-8))

    def auc(self):
        total_auc = 0
        for user in range(self.num_users):
            _, i_range = self.sparse_ratings[user].nonzero()
            if i_range.size == 0:
                continue
            mask = np.zeros(self.num_items).astype(bool)
            mask[i_range] = True

            recs = self.U[user] @ self.V.T
            recs_pos = recs[mask][:, np.newaxis]
            recs_neg = recs[~mask]
            num_pos = mask.sum()
            num_neg = self.num_items - num_pos
            auc = (recs_pos > recs_neg).sum() / (num_pos * num_neg * self.num_users)
            total_auc += auc
        return total_auc

    def fit(self, ratings):
        self.ratings = ratings
        self.ratings["unique_movie"] = self.ratings.groupby("movie_id").ngroup()
        self.ratings["unique_user"] = self.ratings.groupby("user_id").ngroup()

        self.n = self.ratings.shape[0]
        self.users, self.items = np.unique(self.ratings["user_id"].values), np.unique(self.ratings["movie_id"].values)
        self.num_users, self.num_items = len(self.users), len(self.items)

        self.U = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_users, self.latent_size))
        self.V = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_items, self.latent_size))

        self.sparse_ratings = self._get_csr_matrix(self.ratings)
        ui_pairs = self._get_ui_pairs()

        for epoch in tqdm(range(self.epochs)):
            ui_pairs_perm = np.random.permutation(ui_pairs)
            for u, i in ui_pairs_perm:
                j = None
                while j is None:
                    j_cand = np.random.randint(low=0, high=self.num_items)
                    j = j_cand if self.sparse_ratings[u, j_cand] == 0 else None

                x_ui = self.U[u] @ self.V[i]
                x_uj = self.U[u] @ self.V[j]
                x_uij = x_ui - x_uj

                sigm = self.sigmoid(x_uij)

                self.U[u] = self.U[u] + self.lr * (sigm * (self.V[i] - self.V[j]) + self.lambd * self.U[u])
                self.V[i] = self.V[i] + self.lr * (sigm * self.U[u] + self.lambd * self.V[i])
                self.V[j] = self.V[j] + self.lr * (-sigm * self.U[u] + self.lambd * self.V[j])

            auc = self.auc()
            print(f"epoch {epoch + 1}: AUC = {auc:.4f}")


class WARP:
    def __init__(self, latent_size=64, lambd=1e-5, learning_rate=1e-3, max_neg=10, epochs=10):
        self.lambd = lambd
        self.latent_size = latent_size
        self.lr = learning_rate
        self.epochs = epochs
        self.max_neg = max_neg

    def _get_ui_pairs(self):
        data = []
        for user in range(self.num_users):
            _, i_range = self.sparse_ratings[user].nonzero()
            data.extend([[user, i] for i in i_range])
        return np.array(data)

    def _get_csr_matrix(self, dense_matrix):
        implicit_ratings = dense_matrix.loc[(dense_matrix["rating"] >= 4)]

        users = implicit_ratings["unique_user"]
        movies = implicit_ratings["unique_movie"]
        user_item = sp.coo_matrix((np.ones_like(users), (users, movies)))
        user_item_csr = user_item.tocsr()
        return user_item_csr

    def auc(self):
        total_auc = 0
        for user in range(self.num_users):
            _, i_range = self.sparse_ratings[user].nonzero()
            if i_range.size == 0:
                continue
            mask = np.zeros(self.num_items).astype(bool)
            mask[i_range] = True

            recs = self.U[user] @ self.V.T
            recs_pos = recs[mask][:, np.newaxis]
            recs_neg = recs[~mask]
            num_pos = mask.sum()
            num_neg = self.num_items - num_pos
            auc = (recs_pos > recs_neg).sum() / (num_pos * num_neg * self.num_users)
            total_auc += auc
        return total_auc

    def fit(self, ratings):
        self.ratings = ratings
        self.ratings["unique_movie"] = self.ratings.groupby("movie_id").ngroup()
        self.ratings["unique_user"] = self.ratings.groupby("user_id").ngroup()

        self.n = self.ratings.shape[0]
        self.users, self.items = np.unique(self.ratings["user_id"].values), np.unique(self.ratings["movie_id"].values)
        self.num_users, self.num_items = len(self.users), len(self.items)

        self.U = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_users, self.latent_size))
        self.V = np.random.uniform(low=0, high=1 / self.latent_size ** 0.5, size=(self.num_items, self.latent_size))

        self.sparse_ratings = self._get_csr_matrix(self.ratings)
        ui_pairs = self._get_ui_pairs()

        for epoch in range(self.epochs):
            ui_pairs_perm = np.random.permutation(ui_pairs)
            for u, i in tqdm(ui_pairs_perm):
                f_i = self.U[u] @ self.V[i]

                j = None
                for Q in range(1, self.max_neg + 1):
                    while j is None:
                        j_cand = np.random.randint(low=0, high=self.num_items)
                        j = j_cand if self.sparse_ratings[u, j_cand] == 0 else None
                    f_j = self.U[u] @ self.V[j]

                    if 1 - f_i + f_j > 0:
                        grad_u = self.V[i] - self.V[j]
                        if np.linalg.norm(grad_u) > 1:
                            grad_u = np.clip(grad_u, -0.5, 0.5)

                        grad_v = self.U[u]
                        if np.linalg.norm(grad_v) > 1:
                            grad_v = np.clip(grad_v, -0.5, 0.5)

                        loss = np.log(np.floor(self.max_neg / Q)) * (1 - f_i + f_j)
                        self.U[u] = self.U[u] + self.lr * (loss * grad_u + self.lambd * self.U[u])
                        self.V[i] = self.V[i] + self.lr * (loss * grad_v + self.lambd * self.V[i])
                        self.V[j] = self.V[j] + self.lr * (-loss * grad_v + self.lambd * self.V[j])

            auc = self.auc()
            print(f"epoch {epoch + 1}: AUC = {auc:.4f}")
