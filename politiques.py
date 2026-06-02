import pymc as pm 
import pymc.math as pmm 
import numpy as np
import arviz as az
#from env import Individu
from mdp_jax import backward_induction
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from tools import truncated_normal
from individu import Individu




class Politique:
    def __init__(self,known_parameter):
        self.known_parameter = known_parameter
        """
        Base class for decision policies.

        known_parameter : dict
            Contains known global parameters (sigma, grid bounds, etc.)
        """
    
    def choisir_action(self, individu: Individu,alpha,phi,x_min,x_max,step_gap,cout,sigma, *args, **kwargs):
        
        """
        Chooses an action using backward induction (dynamic programming).

        Parameters
        ----------
        individu : Individu
            The individual whose state is used.
        alpha : list
            Estimated alpha parameters.
        phi : float
            Estimated phi for this individual.
        x_min, x_max : float
            Bounds of the discretized state grid.
        step_gap : float
            Grid resolution.
        cout : list
            Action costs.
        sigma : float
            Noise level.

        Returns
        -------
        int
            The action minimizing the value function.
        """
        
        V=None
        if V is None:
            V = backward_induction(
                jnp.array(alpha),
                jnp.array([sigma]*len(alpha)),
                jnp.array(cout),
                phi,
                x_min,
                x_max,
                step_gap,
                120,
                show_progress=False,
                return_all_timesteps=True,
            )
            V = jax.device_get(V)  # we return to CPU only once
        else:
            pass

        #Prevents grid overruns
        state_prev_clipped = np.clip(individu.state_prev, x_min, x_max - 1e-12)
        state_curr_clipped = np.clip(individu.state_curr,   x_min, x_max - 1e-12)

        state_prev = int((state_prev_clipped - x_min) / step_gap)
        state_curr   = int((state_curr_clipped - x_min) / step_gap)

        if individu.state_curr >= x_max or individu.state_curr <=x_min:
            print("ALERT")
            return 1
        values = V[:, individu.step_count, state_curr, state_prev]

        print("values", values)
        action = int(np.argmin(values))
        return action

                

    def alpha_known(self):
        return self.alpha_known
    def need_historique(self):
        return self.need_historique


    def estimation_parametre(self,alpha,individu_actif,historique):
        raise NotImplementedError("implement the estimation of phi and alpha")



class PolitiqueOracleOnline(Politique):
    def __init__(self,known_parameter):
        """
        Oracle policy: it always uses the true underlying parameters.
        Useful as a performance benchmark.
        """
        super().__init__(known_parameter)
        self.alpha_known = True
        self.need_historique = True

    def estimation_parametre(self,alpha_,active_individuals,historique=None):
        """
        Returns the true alpha and true phi for each individual.
        No estimation is performed.
        """
        liste_phi = {individu.id: individu.true_param['phi'] for individu in active_individuals}         
        return alpha_ ,liste_phi


class PolitiqueNothing(Politique):
    def __init__(self,known_parameter):
        """
        Baseline policy: always chooses action 0.
        Useful as a naive control group.
        """
        super().__init__(known_parameter)
        self.alpha_known = True
        self.need_historique = True

    def estimation_parametre(self,alpha_,active_individuals,historique=None):
        """
        Same as Oracle: returns true phi, does not estimate anything.
        """
        liste_phi = {individu.id: individu.true_param['phi'] for individu in active_individuals}         
        return alpha_ ,liste_phi

    def choisir_action(self, individu: Individu,alpha,phi,x_min,x_max,step_gap,cout,sigma, *args, **kwargs):
        """Always returns action 0."""
        return 0



class PolitiqueMCMCpymc(Politique):
    def __init__(self,known_parameter):
        """
        Bayesian policy using PyMC to estimate alpha and phi.
        """
        super().__init__(known_parameter)
        self.alpha_known = False
        self.need_historique = True


    def estimation_parametre(self, alpha, active_individuals, historique=None):
        """
        Estimates alpha and phi using a PyMC hierarchical model.

        - alpha is shared across individuals
        - phi is individual-specific
        - Observations come from the historical dataset
        """
        # Create sets of unique patient IDs
        id_patient_historique = set(historique["patient"])
        id_patient_actif = {individu.id for individu in active_individuals}
        intersection = id_patient_actif & id_patient_historique
        # Case where the histories are empty (no data)
        if not intersection:

            return alpha, {individu.id: 0 for individu in active_individuals}

        else:
            
            unique_patient_ids = id_patient_actif.union(id_patient_historique)
            
            # Mapping patient names to a unique numeric index
            patients_id_to_idx = {name: idx for idx, name in enumerate(unique_patient_ids)}

            # Replace patient names in history with their indices
            patient_indices = np.array([patients_id_to_idx[name] for name in historique["patient"]])

            # Model settings
            sigma = self.known_parameter['sigma']
            nbr_de_patient = len(unique_patient_ids)
            nbr_daction = len(alpha)

            # History of actions and observations
            s_t = np.array(historique["XT+1"]) 
            s_tm1 = np.array(historique["XT"])
            s_tm2 = np.array(historique["XT-1"])
            a_tm1 = np.array(historique["action"])

            with pm.Model() as model:
                # alpha1 ~ TruncatedNormal(0, 100)
                # alpha = [0, alpha1[0], alpha1[1], ...]
                # phi[i] ~ TruncatedNormal(0, 100)

                # mu_t = (1 - alpha[a_t] + phi[i]) * x_t
                #        - (1 - alpha[a_t]) * phi[i] * x_{t-1}

                # y_t ~ Normal(mu_t, sigma)

                alpha1 = pm.TruncatedNormal(
                    "alpha1",
                    mu=0,
                    sigma=100,
                    lower=0,
                    upper=1,
                    shape=nbr_daction - 1
                )
                # Build full alpha
                alpha = pm.Deterministic("alpha", pmm.concatenate([[0.0], alpha1]))
                # Define phi for each unique individual (patients)
                phi = pm.TruncatedNormal(
                    "phi",
                    mu=0,
                    sigma=100,
                    lower=-1,
                    upper=1,
                    shape=nbr_de_patient  # A phi for each unique patient
                )

                # Calculation of the mu term
                mu = (1 - alpha[a_tm1] + phi[patient_indices]) * s_tm1 - (1 - alpha[a_tm1]) * phi[patient_indices] * s_tm2

                # Define the likelihood
                y = pm.Normal("y", mu=mu, sigma=sigma, observed=s_t)

                # Perform MCMC sampling
                trace = pm.sample(1000, tune=500, target_accept=0.9, progressbar=False, random_seed=None)

                # Access the posterior samples
                posterior = trace.posterior

            # Number of chains and draws
            n_chains = posterior.sizes["chain"]
            n_draws = posterior.sizes["draw"]

            rng = np.random.default_rng()  # Using rng as a generator
            # Select a random draw from the chains and draws
            chain_idx = rng.integers(n_chains)
            draw_idx = rng.integers(n_draws)

            alpha_sample = posterior["alpha"].isel(chain=chain_idx, draw=draw_idx).values
            phi_sample = posterior["phi"].isel(chain=chain_idx, draw=draw_idx).values   

            #  Create the phi dictionary with each individual's ID
            phi_MCMC_dict = {}
            for individu in active_individuals:
               # Check if the individual is in the sample
                idx = patients_id_to_idx.get(individu.id, None)
                if idx is not None:
                    # Assign the calculated value of phi for this individual
                    phi_MCMC_dict[individu.id] = phi_sample[idx]
                else:
                    # Otherwise, set to 0
                    phi_MCMC_dict[individu.id] = 0
            # Return alpha and phi as a dictionary
            return alpha_sample.tolist(), phi_MCMC_dict



class Politiquefrequentistescipy(Politique):
    def __init__(self, known_parameter):
        """
        Frequentist policy using SciPy's L-BFGS-B optimizer
        to estimate alpha and phi by minimizing squared prediction error.
        """
        super().__init__(known_parameter)
        self.alpha_known = False
        self.need_historique = True



    def estimation_parametre(self,alpha_init, active_individuals, historique):
        alpha_init[1:] = [1.0] * (len(alpha_init) - 1)
         # Create sets of unique patient IDs
        id_patient_historique = set(historique["patient"])
        id_patient_actif = {individu.id for individu in active_individuals}
        intersection = id_patient_actif & id_patient_historique

        # Case where the histories are empty (no data)
        if not intersection:

            return alpha_init, {individu.id: 0 for individu in active_individuals}
        else:
            unique_patient_ids = list(id_patient_actif.union(id_patient_historique))
            # Extract historical data and prepare the data
            s_t = np.array(historique["XT+1"])  # Status at t+1
            s_tm1 = np.array(historique["XT"])  # Status at t
            s_tm2 = np.array(historique["XT-1"])  # Status at t-1
            a_tm1 = np.array(historique["action"])  #Actions taken at  t
            patient_id = historique["patient"]  # patient identifiers

            unique_actions = sorted(set(a_tm1))
            actions_to_optimize = [a for a in unique_actions if a != 0]



            def cout(params, s_t, s_tm1, s_tm2, a_tm1, patient_id,
                unique_patient_ids, actions_to_optimize, alpha_init):
                """
                Computes the sum of squared prediction errors for the model:

                    x_{t+1} ≈ (1 - alpha[a] + phi[i]) * x_t
                              - (1 - alpha[a]) * phi[i] * x_{t-1}

                Parameters are:
                    - alpha for actions != 0
                    - phi for each patient
                """

               # --- rebuild alpha ---
                alpha = np.copy(alpha_init)

                n_alpha_opt = len(actions_to_optimize)

                # Update only the optimized actions
                for i, action in enumerate(actions_to_optimize):
                    alpha[action] = params[i]

                # --- rebuild phi ---
                phi_offset = n_alpha_opt
                phi = {
                    unique_patient_ids[i]: params[phi_offset + i]
                    for i in range(len(unique_patient_ids))
                }

                # --- predictions ---
                pred = np.zeros_like(s_t)

                for t in range(len(s_t)):
                    action = a_tm1[t]
                    patient = patient_id[t]

                    alpha_value = alpha[action] # alpha[0] automatically remains 0
                    phi_value = phi[patient]

                    pred[t] = (
                        (1 - alpha_value + phi_value) * s_tm1[t]
                        - (1 - alpha_value) * phi_value * s_tm2[t]
                    )

                erreur = np.sum((s_t - pred) ** 2)
                return erreur


            # initial alpha (copy)
            alpha = np.copy(alpha_init)

            # alpha to be optimized (excluding alpha[0])
            alpha_init_opt = np.array([alpha_init[a] for a in actions_to_optimize])

            # initial phi
            unique_patient_ids = list(set(patient_id))
            phi_init = np.zeros(len(unique_patient_ids))

            # parameter vector
            params_init = np.concatenate([alpha_init_opt, phi_init])
          
            
            # Define the ranges for alpha and phi (for example, alpha in [0, 1], phi in [-1, 1])

            bounds_alpha = [(0, 1)] * len(actions_to_optimize)
            bounds_phi = [(-1, 1)] * len(unique_patient_ids)  # boundaries for phi (each patient)
            bounds = bounds_alpha + bounds_phi

            # Minimizing the cost function (L-BFGS-B is often used for problems with bounds)
            resultat = minimize(
                cout,
                params_init,
                args=(s_t, s_tm1, s_tm2, a_tm1, patient_id,
                    unique_patient_ids, actions_to_optimize, alpha_init),
                method='L-BFGS-B',
                bounds=bounds
            )




            alpha_opt = np.copy(alpha_init)

            for i, action in enumerate(actions_to_optimize):
                alpha_opt[action] = resultat.x[i]


            # Entire patient population
            patients_historiques = set(historique["patient"])
            active_patients = {individu.id for individu in active_individuals}
            tous_les_patients = list(patients_historiques.union(active_patients))

            # Estimation for existing patients
            phi_estimes = {}
            offset = len(actions_to_optimize)
            for i, pid in enumerate(unique_patient_ids):
                phi_estimes[pid] = float(resultat.x[offset + i])

            # Add assets missing from the history
            for pid in active_patients:
                if pid not in phi_estimes:
                    phi_estimes[pid] = 0.0

            #print("Estimated phi values (key = patient history)", phi_estimes)
            #print("active patients:", active_patients)
            #print("Estimated alpha values (key = patient history) :", alpha_opt)

            return alpha_opt, phi_estimes



   






class PolitiqueMCMCGibs(Politique):
    def __init__(self,known_parameter):
        """
        Custom Gibbs sampler for estimating alpha and phi.
        """
        super().__init__(known_parameter)
        self.alpha_known = False
        self.need_historique = True


    def estimation_parametre(self, alpha, active_individuals, historique=None):

         # Create sets of unique patient IDs
        id_patient_historique = set(historique["patient"])
        id_patient_actif = {individu.id for individu in active_individuals}
        intersection = id_patient_actif & id_patient_historique

        # Case where the histories are empty (no data)
        if not intersection:
            return alpha, {individu.id: np.random.uniform(-1, 1,) for individu in active_individuals}    
        
        else:
            unique_patient_ids = sorted(id_patient_actif.union(id_patient_historique))
            sigma = self.known_parameter['sigma']
            nbr_daction = len(alpha)

            # Extract historical data and prepare the data
            s_t = np.array(historique["XT+1"])  # Status at t+1
            s_tm1 = np.array(historique["XT"])  # Status at t
            s_tm2 = np.array(historique["XT-1"])  # Status at t-1
            a_tm1 = np.array(historique["action"])  #Actions taken at  t
            patient_id = historique["patient"]  # patient identifiers

             # Mapping patient → index
            patient_to_idx = {p: i for i, p in enumerate(unique_patient_ids)}
            patient_idx = np.array([patient_to_idx[p] for p in patient_id])

            # Pre-calculation of masks by action
            mask_by_action = {a: (a_tm1 == a) for a in range(nbr_daction)}






            def sample_alpha(a, Phi, s_t, s_tm1, s_tm2, mask, patient_idx, sigma):
                """
                Samples alpha[a] from its conditional distribution.
                Uses a truncated normal approximation.
                """
                if not np.any(mask[a]):
                    return np.random.uniform(0, 1)

                s1, s2, st = s_tm1[mask[a]], s_tm2[mask[a]], s_t[mask[a]]
                phi_vals = Phi[patient_idx[mask[a]]]

                denom = np.sum((s1 - phi_vals * s2) ** 2)
                if denom == 0:
                    return np.random.uniform(0, 1)
                
                else:
                    sigma2 = (sigma**2) / denom
                    mean = sigma2 * np.sum((st - phi_vals * s1) * (s1 - phi_vals * s2)) / (sigma**2)
                    res =1-np.random.normal(mean, np.sqrt(sigma2))

                    compteur=0
                    while res > 1 or res < 0:
                        compteur+=1
                        res = 1-np.random.normal(mean, np.sqrt(sigma2))
                        if compteur==1000:
                            res = max(0,min(1,res))
                    return res
                



            def sample_phi(Alpha, Phi, s_t, s_tm1, s_tm2, a_tm1, patient_idx, sigma):
                """
                Samples phi[i] for each patient from its conditional distribution.
                """
                # coeff(a) = 1 - Alpha[a_tm1] for each observation
                coeff = (1.0 - Alpha[a_tm1])                  # shape (n_obs,)
                d1 = (s_tm1 - coeff * s_tm2)                  # shape (n_obs,)
                num = (s_t  - coeff * s_tm1) * d1             # shape (n_obs,)
                denom = d1**2                                 # shape (n_obs,)

                # Aggregation by patient
                n_patients = len(Phi)
                denom_sum = np.bincount(patient_idx, weights=denom, minlength=n_patients)
                num_sum   = np.bincount(patient_idx, weights=num,   minlength=n_patients)

                # Parameters of the conditional distribution
                nonzero = denom_sum > 0
                mean    = np.zeros(n_patients, dtype=float)
                sigma2  = np.zeros(n_patients, dtype=float)

                sigma2[nonzero] = (sigma**2) / denom_sum[nonzero]
                mean[nonzero]   = (sigma2[nonzero] * num_sum[nonzero]) / (sigma**2)

                # Truncated normal sampling for patients with observed data
                res = np.empty_like(Phi, dtype=float)
                if np.any(nonzero):
                    res[nonzero] = truncated_normal(mean[nonzero], np.sqrt(sigma2[nonzero]),
                                                    lower=-1.0, upper=1.0)

                # # Patients without observations → uniform distribution
                if np.any(~nonzero):
                    res[~nonzero] = np.random.uniform(-1.0, 1.0, size=(~nonzero).sum())

                return res


            def gibbs_sampling(
                nbr_daction,
                unique_patient_ids,
                s_t, s_tm1, s_tm2, a_tm1,
                mask_by_action, patient_idx,
                sigma,
                max_iterations=1000,
                nbr_chaine=10,
                burn=500
            ):
                """
                Runs multiple Gibbs chains in parallel.

                Returns
                -------
                Alpha[0] : array
                    Posterior sample of alpha.
                Phi_dict : dict
                    Posterior sample of phi for each patient.
                """
                assert burn < max_iterations, "burn doit être < max_iterations"
                draws = max_iterations - burn

                Alpha = []
                Phi   = []

                alpha_samples = []
                phi_samples   = []

                # Chain initialization
                for j in range(nbr_chaine):
                    alpha_j = np.random.uniform(0, 1, nbr_daction)
                    alpha_j[0] = 0.0
                    phi_j = np.random.uniform(-1, 1, len(unique_patient_ids))

                    Alpha.append(alpha_j)
                    Phi.append(phi_j)

                    alpha_samples.append(np.zeros((draws, nbr_daction)))
                    phi_samples.append(np.zeros((draws, len(unique_patient_ids))))

                # Gibbs
                for i in range(max_iterations):
                    for j in range(nbr_chaine):

                        Alpha[j][0] = 0.0  # fixed parameter

                        #Alpha Update
                        for a in range(1, nbr_daction):
                            Alpha[j][a] = sample_alpha(
                                a, Phi[j],
                                s_t, s_tm1, s_tm2,
                                mask_by_action, patient_idx,
                                sigma
                            )

                        # phi update
                        Phi[j] = sample_phi(
                            Alpha[j], Phi[j],
                            s_t, s_tm1, s_tm2,
                            a_tm1, patient_idx,
                            sigma
                        )

                        # Recording after burn-in
                        if i >= burn:
                            idx = i - burn
                            alpha_samples[j][idx, :] = Alpha[j]
                            phi_samples[j][idx, :]   = Phi[j]

                # Array stacking (chains, draws, params)
                alpha_samples = np.stack(alpha_samples, axis=0)  # (C, D, A)
                phi_samples   = np.stack(phi_samples,   axis=0)  # (C, D, P)

                # We remove alpha[0], which is fixed
                alpha_for_arviz = alpha_samples[:, :, 1:]  # We keep actions 1 through A-1

                posterior = {
                    "alpha": alpha_for_arviz,
                    "phi": phi_samples,
                }

                idata = az.from_dict(posterior=posterior)
                rhat = az.rhat(idata)
                ess  = az.ess(idata)

                seuils = {"Rhat": 1.05, "ESS": 10}
                verdicts = {}

                for var in rhat.data_vars:
                    rhat_vals = rhat[var].values.flatten()
                    ess_vals  = ess[var].values.flatten()

                    verdicts[var] = []
                    for idx_p, (r, e) in enumerate(zip(rhat_vals, ess_vals)):
                        ok = (r < seuils["Rhat"]) and (e > seuils["ESS"])
                        verdicts[var].append({
                            "param_index": int(idx_p),
                            "Rhat": float(r),
                            "ESS": float(e),
                            "Converged": ok,
                        })

                for var, params in verdicts.items():
                    for p in params:
                        if not p["Converged"]:
                            print("non convergence")

                # Reconstruction of the Phi Dictionary for Patients
                Phi_dict = {p: Phi[0][patient_to_idx[p]] for p in unique_patient_ids}

                return Alpha[0], Phi_dict



            alphatest, list_phitest = gibbs_sampling(nbr_daction, unique_patient_ids, s_t, s_tm1, s_tm2, a_tm1,mask_by_action,patient_idx, sigma)
            return alphatest , list_phitest



#############################


# class Agent():
#     def __init__(self, agent_cls, parametre_connu, n_envs =1, seed):
#         self.parametre_connu = parametre_connu
#         self.rng = np.random.RandomState(seed)
#         self.n_envs = n_envs
#         seed = self.rng.randint(1,2**32)
#         self.learning_algo = agent_cls(parametre_connu, seed)

#     def learn_1step(self, n_iterations = 100):
#         alpha_ = self.alpha
#         hist = self.hist
#         individus_actifs = self.individus_actifs
#         alpha, liste_phi = self.learning_algo.estimation_parametre(alpha_, 
#                                                                    individus_actifs, 
#                                                                    hist)
#         self.estimations_alpha.append(alpha)
#         self.estimations_phi.append(liste_phi)

#     def predict(self, observation):
#         # TODO: alpha ? phi ?
#         nouveaux_actifs = []
#         actions = np.zeros(self.n_envs)
#         for f in range(self.n_envs):
#             if observation[f] != -999: 
#                 actions[f] = self.learning_algo.choisir_action(
#                     individu, alpha, phi = liste_phi[f],
#                     x_min=self.x_min, x_max=self.x_max,
#                     ecart_de_pas=self.ecart_de_pas, cout=self.cout,
#                     sigma=self.sigma, seed=self.rng.randint(1,2**32)
#                     )
#                 nouveaux_actifs.append(f)
#                 self.hist["action"].append(actions[f])
#                 self.hist["patient"].append(f)
#                 self.hist["XT+1"].append(etat[0])
#                 self.hist["XT"].append(etat[1])
#                 self.hist["XT-1"].append(etat[2])
#                 self.hist["temps"].append(self.temps_global)

#         self.individus_actifs = nouveaux_actifs
