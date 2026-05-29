import pymc as pm 
import pymc.math as pmm 
import numpy as np
import arviz as az
from env import Individu
from mdp_jax import backward_induction
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from tools import truncated_normal

class Politique:
    def __init__(self,parametre_connu):
        self.parametre_connu = parametre_connu

    def choisir_action(self, individu: Individu,alpha,phi,x_min,x_max,ecart_de_pas,cout,sigma, *args, **kwargs):
        
        V=None
        if V is None:
            #print(f"Calcul backward_induction pour phi={phi}, alpha={alpha_tuple}")
            V = backward_induction(
                jnp.array(alpha),
                jnp.array([sigma, sigma]),
                jnp.array(cout),
                phi,
                x_min,
                x_max,
                ecart_de_pas,
                120,
                show_progress=False,
                return_all_timesteps=True,
            )
            V = jax.device_get(V)  # on ramène sur CPU une seule fois
            #self._save_to_cache(cache_key, V)
        else:
            pass
        #t1 = time.time()
        #print("Temps exécution backward_induction:", t1- t0)      
        #     
        # Empêche les dépassements de grille
        etat_prev_clipped = np.clip(individu.etat_precedent, x_min, x_max - 1e-12)
        etat_curr_clipped = np.clip(individu.etat_courant,   x_min, x_max - 1e-12)

        etat_precedent = int((etat_prev_clipped - x_min) / ecart_de_pas)
        etat_courant   = int((etat_curr_clipped - x_min) / ecart_de_pas)

        ##etat_precedent = int((individu.etat_precedent-x_min)/ecart_de_pas)
        ##etat_courant = int((individu.etat_courant-x_min)/ecart_de_pas)
        if individu.etat_courant >= x_max or individu.etat_courant <=x_min:
            action = int(np.argmax(alpha))
            return action
        valeurs = V[:, individu.step_count, etat_courant, etat_precedent]
        #valeur1, valeur2 = valeurs[0], valeurs[1]

        #print("valeur", valeurs)
        action = int(np.argmin(valeurs))
        return action

 

        

    def alpha_connue(self):
        return self.alpha_connue
    def besoin_historique(self):
        return self.besoin_historique


    def estimation_parametre(self,alpha,individu_actif,historique):
        raise NotImplementedError("implémenter l'estimation des phi et alpha")



class PolitiqueOracleOnline(Politique):
    def __init__(self,parametre_connu):
        super().__init__(parametre_connu)
        self.alpha_connue = True
        self.besoin_historique = True

    def estimation_parametre(self,alpha_,individus_actifs,historique=None):
        liste_phi = {individu.id: individu.true_param['phi'] for individu in individus_actifs}         
        return alpha_ ,liste_phi




class PolitiqueMCMCpymc(Politique):
    def __init__(self,parametre_connu):
        super().__init__(parametre_connu)
        self.alpha_connue = False
        self.besoin_historique = True


    def estimation_parametre(self, alpha, individus_actifs, historique=None):
        # Créer des ensembles d'IDs uniques des patients
        id_patient_historique = set(historique["patient"])
        id_patient_actif = {individu.id for individu in individus_actifs}
        intersection = id_patient_actif & id_patient_historique
        # Cas où les historiques sont vides (aucune donnée)
        if not intersection:

            return alpha, {individu.id: 0 for individu in individus_actifs}

        else:
            
            ids_patient_uniques = id_patient_actif.union(id_patient_historique)
            
            # Mappage des noms des patients vers un indice numérique unique
            patients_id_to_idx = {name: idx for idx, name in enumerate(ids_patient_uniques)}

            # Remplacer les noms des patients dans historique par leurs indices
            patient_indices = np.array([patients_id_to_idx[name] for name in historique["patient"]])

            # Paramètres du modèle
            sigma = self.parametre_connu['sigma']
            nbr_de_patient = len(ids_patient_uniques)
            nbr_daction = len(alpha)

            # Historique des actions et observations
            s_t = np.array(historique["XT+1"]) 
            s_tm1 = np.array(historique["XT"])
            s_tm2 = np.array(historique["XT-1"])
            a_tm1 = np.array(historique["action"])

            with pm.Model() as model:
                # Définir alpha1 avec une distribution TruncatedNormal
                alpha1 = pm.TruncatedNormal(
                    "alpha1",
                    mu=0,
                    sigma=100,
                    lower=0,
                    upper=1,
                    shape=nbr_daction - 1
                )

                # Construire alpha complet
                alpha = pm.Deterministic("alpha", pmm.concatenate([[0.0], alpha1]))

                # Définir phi pour chaque individu unique (patients)
                phi = pm.TruncatedNormal(
                    "phi",
                    mu=0,
                    sigma=100,
                    lower=-1,
                    upper=1,
                    shape=nbr_de_patient  # Un phi pour chaque patient unique
                )

                # Calcul du terme mu
                mu = (1 - alpha[a_tm1] + phi[patient_indices]) * s_tm1 - (1 - alpha[a_tm1]) * phi[patient_indices] * s_tm2

                # Définir la vraisemblance
                y = pm.Normal("y", mu=mu, sigma=sigma, observed=s_t)

                # Effectuer l'échantillonnage MCMC
                trace = pm.sample(1000, tune=500, target_accept=0.9, progressbar=False, random_seed=None)

                # Accéder aux échantillons postérieurs
                posterior = trace.posterior

            # Nombre de chaînes et de tirages
            n_chains = posterior.sizes["chain"]
            n_draws = posterior.sizes["draw"]

            rng = np.random.default_rng()  # Utilisation de rng comme générateur
            # Choisir un tirage aléatoire parmi les chaînes et les tirages
            chain_idx = rng.integers(n_chains)
            draw_idx = rng.integers(n_draws)

            alpha_sample = posterior["alpha"].isel(chain=chain_idx, draw=draw_idx).values
            phi_sample = posterior["phi"].isel(chain=chain_idx, draw=draw_idx).values


            # Debug: vérifier la forme des échantillons
           # print(f"Shape of alpha_sample: {alpha_sample.shape}")
           # print(f"Shape of phi_sample: {phi_sample.shape}")          

            # Créer le dictionnaire de phi avec l'id de chaque individu
            phi_MCMC_dict = {}
            for individu in individus_actifs:
                # Vérifier si l'individu est dans l'échantillon
                idx = patients_id_to_idx.get(individu.id, None)
                if idx is not None:
                    # Assigner la valeur de phi tirée pour cet individu
                    phi_MCMC_dict[individu.id] = phi_sample[idx]
                else:
                    # Sinon, assigner 0
                    phi_MCMC_dict[individu.id] = 0
            # Retourner alpha et phi sous forme de dictionnaire
            return alpha_sample.tolist(), phi_MCMC_dict



class Politiquefrequentistescipy(Politique):
    def __init__(self, parametre_connu):
        super().__init__(parametre_connu)
        self.alpha_connue = False
        self.besoin_historique = True



    def estimation_parametre(self,alpha_init, individus_actifs, historique):
        alpha_init[1]=0.1
        # Créer des ensembles d'IDs uniques des patients
        id_patient_historique = set(historique["patient"])
        id_patient_actif = {individu.id for individu in individus_actifs}
        intersection = id_patient_actif & id_patient_historique

        # Cas où les historiques sont vides (aucune donnée)
        if not intersection:

            return alpha_init, {individu.id: 0 for individu in individus_actifs}
        else:
            ids_patient_uniques = list(id_patient_actif.union(id_patient_historique))
            # Extraire les historiques et préparer les données
            s_t = np.array(historique["XT+1"])  # état à t+1
            s_tm1 = np.array(historique["XT"])  # état à t
            s_tm2 = np.array(historique["XT-1"])  # état à t-1
            a_tm1 = np.array(historique["action"])  # action à t-1
            patient_id = historique["patient"]  # identifiants des patients


            # Fonction de coût à minimiser
            def cout(params, s_t, s_tm1, s_tm2, a_tm1, patient_id, ids_patient_uniques, alpha_init):
                # Extraire alpha et phi des paramètres optimisés
                alpha = np.copy(alpha_init)  # On garde alpha[0] constant (alpha[0] n'est pas optimisé)
                phi = {ids_patient_uniques[i]: params[1 + i] for i in range(len(ids_patient_uniques))}
                
                # Affecter alpha[1] à alpha optimisé
                alpha[1:] = params[0:len(alpha) - 1]
                
                # Prédictions : 
                pred = np.zeros_like(s_t)
                for t in range(len(s_t)):
                    # Récupérer l'action de a_tm1 et l'ID du patient
                    action = a_tm1[t]  # l'action à t-1
                    patient = patient_id[t]  # ID du patient à t
                    
                    # Calculer la prédiction pour ce patient et ce temps
                    phi_value = phi[patient]  # phi pour ce patient
                    alpha_value = alpha[action]  # alpha pour cette action
                    
                    # Calculer la prédiction en fonction de l'équation donnée
                    pred[t] = (1 - alpha_value + phi_value) * s_tm1[t] - (1 - alpha_value) * phi_value * s_tm2[t]
                
                # Erreur quadratique totale
                erreur = np.sum((s_t - pred) ** 2)
                
                return erreur
            

            # Ensemble d'IDs uniques des patients
            ids_patient_uniques = list(set(patient_id))
            
            # Initialisation de phi pour chaque patient (initialiser à zéro ou à une valeur de départ)
            phi_init = np.zeros(len(ids_patient_uniques))
            
            # Initialisation des paramètres à optimiser (alpha[1:], phi)
            params_init = np.concatenate([alpha_init[1:], phi_init])
            
            # Définir les bornes pour alpha et phi (par exemple, alpha dans [0, 1], phi dans [-1, 1])
            bounds_alpha = [(0, 1)] * (len(alpha_init) - 1)  # bornes pour alpha[1:] (on ne touche pas alpha[0])
            bounds_phi = [(-1, 1)] * len(ids_patient_uniques)  # bornes pour phi (chaque patient)
            bounds = bounds_alpha + bounds_phi
            # Minimisation de la fonction de coût (L-BFGS-B est souvent utilisé pour les problèmes avec des bornes)
            resultat = minimize(cout, params_init, args=(s_t, s_tm1, s_tm2, a_tm1, patient_id, ids_patient_uniques, alpha_init),
                                method='L-BFGS-B', bounds=bounds)

            # Récupérer les paramètres optimisés
            alpha_opt = np.array([0] + list(resultat.x[:len(alpha_init) - 1]))  # Ajouter alpha[0] = 0
           # Construire phi uniquement pour les patients uniques optimisés
            phi_opt = {}
            for i, patient_id in enumerate(ids_patient_uniques):
                # retrouver l'individu actif correspondant à ce patient
                for individu in individus_actifs:
                    if individu.id.startswith(patient_id):  # ex: "ind_4_freq" commence par "ind_4"
                        phi_opt[individu.id] = resultat.x[len(alpha_init) - 1 + i]          
            # Ajouter les individus actifs absents de l'historique avec phi = 0
            for individu in individus_actifs:
                if individu.id not in phi_opt:
                    phi_opt[individu.id] = 0.0
            # Retourner les paramètres optimisés
            return alpha_opt, phi_opt


   



class PolitiqueMCMCGibs(Politique):
    def __init__(self,parametre_connu, seed):
        super().__init__(parametre_connu)
        self.alpha_connue = False
        self.besoin_historique = True
        self.rng = np.random.RandomState(seed)

    def estimation_parametre(self, alpha, individus_actifs, historique=None):
        # Créer des ensembles d'IDs uniques des patients
        id_patient_historique = set(historique["patient"])
        id_patient_actif = {individu.id for individu in individus_actifs}
        intersection = id_patient_actif & id_patient_historique

        # Cas où les historiques sont vides (aucune donnée)
        if not intersection:
            return alpha, {individu.id: self.rng.uniform(-1, 1,) for individu in individus_actifs}    
        
        else:
            ids_patient_uniques = sorted(id_patient_actif.union(id_patient_historique))
            sigma = self.parametre_connu['sigma']
            nbr_daction = len(alpha)

            # Extraire les informations historiques
            s_t = np.array(historique["XT+1"])  # état à t+1
            s_tm1 = np.array(historique["XT"])  # état à t
            s_tm2 = np.array(historique["XT-1"])  # état à t-1
            a_tm1 = np.array(historique["action"])  # action à t-1
        
            patient_id = np.array(historique["patient"])

            # Mapping patient → index
            patient_to_idx = {p: i for i, p in enumerate(ids_patient_uniques)}
            patient_idx = np.array([patient_to_idx[p] for p in patient_id])

            # Pré‑calcul des masques par action
            mask_by_action = {a: (a_tm1 == a) for a in range(nbr_daction)}


            def sample_alpha(a, Phi, s_t, s_tm1, s_tm2, mask, patient_idx, sigma, rng):
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
                    res = truncated_normal(0,1, 1- mean, np.sqrt(sigma2), rng)
                    return res
                

            def sample_phi(Alpha, Phi, s_t, s_tm1, s_tm2, a_tm1, patient_idx, sigma, rng):
                # coeff(a) = 1 - Alpha[a_tm1] pour chaque observation
                coeff = (1.0 - Alpha[a_tm1])                  # shape (n_obs,)
                d1 = (s_tm1 - coeff * s_tm2)                  # shape (n_obs,)
                num = (s_t  - coeff * s_tm1) * d1             # shape (n_obs,)
                denom = d1**2                                 # shape (n_obs,)

                # Agrégation par patient
                n_patients = len(Phi)
                denom_sum = np.bincount(patient_idx, weights=denom, minlength=n_patients)
                num_sum   = np.bincount(patient_idx, weights=num,   minlength=n_patients)

                # Paramètres de la conditionnelle
                nonzero = denom_sum > 0
                mean    = np.zeros(n_patients, dtype=float)
                sigma2  = np.zeros(n_patients, dtype=float)

                sigma2[nonzero] = (sigma**2) / denom_sum[nonzero]
                mean[nonzero]   = (sigma2[nonzero] * num_sum[nonzero]) / (sigma**2)

                # Tirage tronqué pour les patients avec données
                res = np.empty_like(Phi, dtype=float)
                if np.any(nonzero):
                    res[nonzero] = truncated_normal(mean[nonzero],
                                                    np.sqrt(sigma2[nonzero]),
                                                    lower=-1.0, upper=1.0, rng=rng)

                # Patients sans observation → uniforme
                if np.any(~nonzero):
                    res[~nonzero] = rng.uniform(-1.0, 1.0, size=(~nonzero).sum())

                return res


            def gibbs_sampling(
                nbr_daction,
                ids_patient_uniques,
                s_t, s_tm1, s_tm2, a_tm1,
                mask_by_action, patient_idx,
                sigma,
                max_iterations=1000,
                nbr_chaine=10,
                burn=500,
            ):
                rng = self.rng
                assert burn < max_iterations, "burn doit être < max_iterations"
                draws = max_iterations - burn

                Alpha = []
                Phi   = []

                alpha_samples = []
                phi_samples   = []

                # Initialisation des chaînes
                for j in range(nbr_chaine):
                    alpha_j = rng.uniform(0, 1, size=nbr_daction)
                    alpha_j[0] = 0.0
                    phi_j = rng.uniform(-1, 1, size=len(ids_patient_uniques))

                    Alpha.append(alpha_j)
                    Phi.append(phi_j)

                    alpha_samples.append(np.zeros((draws, nbr_daction)))
                    phi_samples.append(np.zeros((draws, len(ids_patient_uniques))))

                # Gibbs
                for i in range(max_iterations):
                    for j in range(nbr_chaine):

                        Alpha[j][0] = 0.0  # paramètre fixé

                        # Mise à jour de alpha
                        for a in range(1, nbr_daction):
                            Alpha[j][a] = sample_alpha(
                                a, Phi[j],
                                s_t, s_tm1, s_tm2,
                                mask_by_action, patient_idx,
                                sigma, rng = rng
                            )

                        # Mise à jour de phi
                        Phi[j] = sample_phi(
                            Alpha[j], Phi[j],
                            s_t, s_tm1, s_tm2,
                            a_tm1, patient_idx,
                            sigma,
                            rng = rng
                        )

                        # Enregistrement après burn-in
                        if i >= burn:
                            idx = i - burn
                            alpha_samples[j][idx, :] = Alpha[j]
                            phi_samples[j][idx, :]   = Phi[j]

                # Empilement en array (chains, draws, params)
                alpha_samples = np.stack(alpha_samples, axis=0)  # (C, D, A)
                phi_samples   = np.stack(phi_samples,   axis=0)  # (C, D, P)

                # On enlève alpha[0] qui est fixe
                alpha_for_arviz = alpha_samples[:, :, 1:]  # on garde les actions 1..A-1

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

                # Reconstruction du dictionnaire Phi pour les patients
                Phi_dict = {p: Phi[0][patient_to_idx[p]] for p in ids_patient_uniques}

                return Alpha[0], Phi_dict



            ## 
            alphatest, list_phitest = gibbs_sampling(nbr_daction, ids_patient_uniques, s_t, s_tm1, s_tm2, a_tm1,mask_by_action,patient_idx, sigma)
            ##alphatest, list_phitest =gibbs_sampling(initialisation_alpha,initialisation_phi,s_t, s_tm1, s_tm2, a_tm1, patient_id, sigma)
            return alphatest , list_phitest


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
