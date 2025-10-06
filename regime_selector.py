# regime_selector.py

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.rpcmci import RPCMCI
from tigramite.data_processing import DataFrame

from tigramite.independence_tests.parcorr import ParCorr



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
        n_jobs: int = -1,                 # will be overridden to 1 if deterministic=True
        # common kwargs
        cond_ind_test = None,
        prediction_model = None,
        verbosity: int = -2,              # our class-level verbosity (not Tigramite)
        # NEW knobs
        seed: int = 1234,                 # fixes RNG inside RPCMCI
       
        silent: bool = True,              # mutes Tigramite’s own prints
        verbose: bool = False             # controls *our* progress prints
    ):
        """
        data: array of shape (T, N)
        tau_max: maximum lag
        The rest parameterize RPCMCI.
        """
        # ---- data ----
        self.data = data[:500]  # keep if you want a cap, or remove to use full
        self.T, self.N = self.data.shape
        self.tau_max = tau_max

        # ---- determinism (OS/env & seeds) ----
        self.seed = seed
        

        # ---- wrap in Tigramite DataFrame ----
        self.df = DataFrame(self.data)

        # ---- silence controls ----
        self.silent = silent
        self.verbose = verbose

        # CI test (mute its internal messages)
        self.ci_test = cond_ind_test if cond_ind_test is not None else ParCorr(
            verbosity=0 if self.silent else 1
        )

        # ---- RPCMCI constructor kwargs ----
        self.rpc_ctor_kwargs = dict(
            dataframe        = self.df,
            cond_ind_test    = self.ci_test,
            prediction_model = prediction_model or None,
            seed             = self.seed,                # <<< fixes RPCMCI annealing RNG
            verbosity        = -1 if self.silent else 0  # <<< mutes Tigramite output
        )

        # ---- RPCMCI run() kwargs (NOT constructor kwargs) ----
        # NOTE: do not include 'dataframe', 'cond_ind_test', etc. here; those belong to the constructor.
        self.rpc_run_kwargs = dict(
            switch_thres   = switch_thres,
            num_iterations = num_iterations,
            max_anneal     = max_anneal,
            tau_min        = tau_min,
            tau_max        = tau_max,
            pc_alpha       = pc_alpha,
            alpha_level    = alpha_level,
            n_jobs         = -1 
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
     
        # Construct RPCMCI once per evaluation to ensure clean state/seed
        rpc = RPCMCI(**self.rpc_ctor_kwargs)

       
        res = rpc.run_rpcmci(
                num_regimes     = NK,
                max_transitions = NC,
                **self.rpc_run_kwargs
        )

        if res is None:
            return np.inf, None

        gamma          = res["regimes"]
        causal_results = res["causal_results"]

        # SSE & log-likelihood
        resid_sq = self.compute_resid_sse(gamma, causal_results)
        SSE  = resid_sq.sum()
        Teff = self.T - self.tau_max
        sigma2 = SSE / (Teff * self.N)
        logL   = -0.5 * (Teff * self.N) * (np.log(2 * np.pi * sigma2) + 1)

        # Count params
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

    def find_best(self, grid_NK, grid_NC, aicc_tol: float = 1e-9):
        """
        Loop over grid_NK, grid_NC to find the combo minimizing AICc.
        Returns a dict mapping (NK, NC) → {'aicc':…, 'n_params':…},
        plus the best (score, NK, NC).
        """
        best_score  = np.inf
        best_params = (None, None)

        all_results = {}

        for NK in sorted(grid_NK):
            for NC in sorted(grid_NC):
                aicc, n_params = self.evaluate(NK, NC)

                all_results[(NK, NC)] = {
                    'aicc'     : float(aicc),
                    'n_params' : n_params
                }

                # stable compare + tie-break (prefer smaller NK, then NC)
                is_better = (aicc + aicc_tol) < best_score
                is_tie    = abs(aicc - best_score) <= aicc_tol
                if is_better or (is_tie and (best_params[0] is None or (NK, NC) < best_params)):
                    best_score  = aicc
                    best_params = (NK, NC)

                if self.verbose:
                    print(f"NK={NK}, NC={NC} → AICc={aicc:.6f}, params={n_params}")

        return {
            'all_results': all_results,
            'best': {
                'aicc':   float(best_score),
                'NK':     best_params[0],
                'NC':     best_params[1]
            }
        }
