from tigramite.rpcmci import RPCMCI
from tigramite.data_processing import DataFrame
from tigramite.independence_tests.parcorr import ParCorr
import numpy as np
from joblib import Parallel, delayed

class RegimeAICSelector:
    def __init__(
        self,
        data: np.ndarray,
        tau_max: int = 1,
        # RPCMCI kwargs
        switch_thres: float = 1e-3,
        num_iterations: int = 10,
        max_anneal: int = 10,
        tau_min: int = 1,
        pc_alpha: float = 0.2,
        alpha_level: float = 0.01,
        n_jobs: int = -1,
        # common kwargs
        cond_ind_test = None,
        prediction_model = None,
        verbosity: int = -2
    ):
        """
        data: array of shape (T, N)
        tau_max: maximum lag
        The rest parameterize RPCMCI.
        """
        self.data = data[:500]  # or full data if you prefer
        self.T, self.N = self.data.shape
        self.tau_max = tau_max

        # wrap in Tigramite DataFrame
        self.df = DataFrame(self.data)

        # common
        self.common_kwargs = dict(
            dataframe        = self.df,
            cond_ind_test    = cond_ind_test  or ParCorr(),
            prediction_model = prediction_model or None,
            verbosity        = verbosity,
        )

        # rpcmci‐specific
        self.rpcmci_kwargs = dict(
            switch_thres   = switch_thres,
            num_iterations = num_iterations,
            max_anneal     = max_anneal,
            tau_min        = tau_min,
            tau_max        = tau_max,
            pc_alpha       = pc_alpha,
            alpha_level    = alpha_level,
            n_jobs         = n_jobs
        )

    @staticmethod
    def extract_parents(causal_res_k: dict, tau_max: int):
        """
        Returns parent_dict[j] = list of (i, lag) pairs for Xi_{t-lag} → Xj_t.
        """
        if "parents_dict" in causal_res_k:
            return causal_res_k["parents_dict"]

        graph = causal_res_k["graph"]  # shape (N, N, tau_max+1)
        N = graph.shape[0]
        pdict = {}
        for j in range(N):
            links = []
            for i in range(N):
                for lag in range(1, tau_max+1):
                    if graph[i, j, lag] == "-->":
                        links.append((i, lag))
            pdict[j] = links
        return pdict

    def compute_resid_sse(self, gamma: np.ndarray, causal_results: dict):
        """
        gamma: (NK, T) regime indicator (0/1)
        causal_results: {k: {'graph':…, 'val_matrix':…}, …}
        returns: resid_sq array (NK, T) of per‐regime SSE at each t
        """
        NK, T = gamma.shape
        resid_sq = np.zeros((NK, T))

        for k in range(NK):
            parents = self.extract_parents(causal_results[k], self.tau_max)
            vals    = causal_results[k]["val_matrix"]  # shape (N,N,tau_max+1)

            for t in range(self.tau_max, T):
                if gamma[k, t] < 0.5:
                    continue
                for j in range(self.N):
                    y_true = self.data[t, j]
                    y_hat  = sum(
                        self.data[t - lag, i] * vals[i, j, lag]
                        for (i, lag) in parents[j]
                    )
                    resid_sq[k, t] += (y_true - y_hat) ** 2

        return resid_sq

    def evaluate(self, NK: int, NC: int):
        """
        Run RPCMCI with (NK, NC), compute AICc.
        Returns (aicc, N_para)
        """
        rpc = RPCMCI(**self.common_kwargs)
        res = rpc.run_rpcmci(
            num_regimes     = NK,
            max_transitions = NC,
            **self.rpcmci_kwargs
        )
        if res is None:
            return np.inf, None

        gamma          = res["regimes"]
        causal_results = res["causal_results"]

        # SSE
        resid_sq = self.compute_resid_sse(gamma, causal_results)
        SSE  = resid_sq.sum()
        Teff = self.T - self.tau_max
        sigma2 = SSE / (Teff * self.N)
        logL   = -0.5 * (Teff * self.N) * (np.log(2 * np.pi * sigma2) + 1)

        # count params
        n_betas = sum(
            len(self.extract_parents(causal_results[k], self.tau_max)[j])
            for k in range(NK) for j in range(self.N)
        )
        Npara = n_betas + (NK - 1) * NC

        # AICc
        aicc = (
            -2 * logL
            + 2 * Npara
            + 2 * Npara * (Npara + 1) / (Teff - Npara - 1)
        )
        return aicc, Npara

    def find_best(self, grid_NK, grid_NC):
        """
        Loop over grid_NK, grid_NC to find the combo minimizing AICc.
        Returns a dict mapping (NK, NC) → {'aicc':…, 'n_params':…},
        plus the best (score, NK, NC).
        """
        best_score = np.inf
        best_params = (None, None)

        # this will store every result
        all_results = {}

        for NK in grid_NK:
            for NC in grid_NC:
                aicc, n_params = self.evaluate(NK, NC)

                # save into your dict
                all_results[(NK, NC)] = {
                    'aicc'     : aicc,
                    'n_params' : n_params
                }

                # update best
                if aicc < best_score:
                    best_score  = aicc
                    best_params = (NK, NC)

                print(f"NK={NK}, NC={NC} → AICc={aicc:.2f}, params={n_params}")

        # wrap up
        return {
            'all_results': all_results,
            'best': {
                'aicc':   best_score,
                'NK':     best_params[0],
                'NC':     best_params[1]
            }
        }

    def find_best_parallel(self, grid_NK, grid_NC, n_jobs=None):
        """
        Parallel grid search over (NK, NC).
        Returns same dict as find_best, but computed in parallel.
        """
        # build list of all (NK, NC) pairs
        candidates = [(NK, NC) for NK in grid_NK for NC in grid_NC]

        # helper to run one evaluation
        def eval_pair(pair):
            NK, NC = pair
            aicc, n_params = self.evaluate(NK, NC)
            return (NK, NC, aicc, n_params)

        # dispatch jobs
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_pair)(pair) for pair in candidates
        )

        # collect into dict & pick best
        all_results = {}
        best_score = np.inf
        best_params = (None, None)

        for NK, NC, aicc, n_params in results:
            all_results[(NK, NC)] = {'aicc': aicc, 'n_params': n_params}
            if aicc < best_score:
                best_score = aicc
                best_params = (NK, NC)

            print(f"NK={NK}, NC={NC} → AICc={aicc:.2f}, params={n_params}")

        return {
            'all_results': all_results,
            'best': {
                'aicc': best_score,
                'NK':   best_params[0],
                'NC':   best_params[1]
            }
        }
