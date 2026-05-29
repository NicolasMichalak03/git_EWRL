import numpy as np
import copy
import json
from gym_postop.envs import PostOpEnv

class Individu:
    def __init__(self, id_individu, true_param , sigma=0.1, 
                 etat_initial=0.0,
                 max_steps=119,seed=None):
        self.id = id_individu
        self.true_param = true_param
        self.sigma = sigma
        self.etat_initial = etat_initial
        self.max_steps = max_steps
        self.gymenv = None
        self.reset(int(seed))

    def reset(self, seed):
        self.historique_rewards = []
        self.historique_etats = [self.etat_initial]
        self.historique_actions = []
        self.etat_precedent = self.etat_initial
        self.etat_courant = self.etat_initial
        self.step_count = 0
        self.done = False
        self.seed = seed
        if self.gymenv is not None:
            self.gymenv.reset(self.seed)


    def step(self, action,cout):
        if self.gymenv is None:
            self.gymenv= PostOpEnv(true_param=self.true_param,
                      etat_initial=self.etat_initial,
                      max_steps=self.max_steps,
                      date_enter=0,
                      parametre_connu={'sigma': self.sigma , 'cout': cout})
            self.gymenv.reset(int(self.seed))

        if self.done:
            print("erreur ? Je crois qu'on doit pas avoir de pas si il est déjà terminé")
            return self.etat_courant, True

        self.historique_actions.append(action)

        alpha = self.true_param['alpha'][action]
        #beta = self.true_param['beta'][action]  #a garder si nescessaire
        phi = self.true_param['phi']

        obs, reward, terminated, done, info = self.gymenv.step(action)
        
        nouvel_etat = float(obs[0])

        # Mise à jour des états
        etat_precedent_temporaire =  self.etat_precedent
        self.etat_precedent = self.etat_courant
        self.etat_courant = nouvel_etat
        self.historique_etats.append(nouvel_etat)
        self.historique_rewards.append(reward)
        self.done = done
        # Estimations
            #Ne doit rien faire si on estime les paramètre phi d'un point de vue global
        self.estimer_parametre()
        self.step_count += 1
       
        # print(self.etat_courant)
        return [self.etat_courant , self.etat_precedent, etat_precedent_temporaire], self.done


    def estimer_parametre(self):
        return
    
    def is_done(self):
        return self.done

    def get_etat(self):
        return self.etat_courant

    def get_historique_actions(self):
        return self.historique_actions
    
    def get_historique(self):
        return self.historique_etats
    
    def get_historique_rewards(self):
        return self.historique_rewards



class SimulateurMultiIndividusGlobal:
    def __init__(self,calendrier_arrivees,politiques,parametre_connu,parametre_cacher,
                 historique_offline={"XT+1" : [], "XT" : [], "XT-1" : [], "patient" :[], "action" : [],"temps":[]} ,globalseed=None):
        

        self.x_max = parametre_connu.get('x_max')
        self.x_min= parametre_connu.get('x_min')
        self.ecart_de_pas = parametre_connu.get('ecart_de_pas')
        self.cout = parametre_connu.get('cout')

        self.true_sigma = parametre_connu.get('sigma') or parametre_cacher.get('sigma')
        self.true_alpha = parametre_connu.get('alpha') or parametre_cacher.get('alpha')

              
        """
        calendrier_arrivees : dict {t_arrivee: liste de tuples (id_individu, parametre_cache)}
        """
        self.calendrier = calendrier_arrivees
        self.politiques = politiques
        
        for nom, politique_instance in self.politiques.items():
            setattr(self, f'individus_actifs_{nom}',[])
            setattr(self, f'individus_termines_{nom}',[])

            # Crée dynamiquement les attribut en fonction du nom de la politique
            if politique_instance.besoin_historique:  
                setattr(self, f'historique_{nom}', historique_offline)

            if politique_instance.besoin_historique:  
                setattr(self, f'historique_{nom}', copy.deepcopy(historique_offline))

            if politique_instance.alpha_connue:  
                # Si connais_alpha() renvoie True renvoie la vrai valeur de alpha
                setattr(self, f'alpha_{nom}', self.true_alpha)
            else: #Sinon donne une valeur par défaut 
                setattr(self, f'alpha_{nom}', [0]*len(self.true_alpha))
            
                # === Nouveau : stockage des estimations ===
            setattr(self, f'estimations_alpha_{nom}', [])
            setattr(self, f'estimations_phi_{nom}', [])

        self.rng = np.random.default_rng(globalseed)
        self.temps_global = 0

        ##############
        #self.Historique = historique_offline
        #self._global_id_counter= max(self.Historique["p"], default=0)



        
        #self_estimate_alpha = [0]*nbr_daction       #NOn utilisé a voir après


    #def _get_unique_id(self):
     #   """Renvoie un identifiant unique et incrémente le compteur global"""
      #  self._global_id_counter += 1
       # return self._global_id_counter


    def step_global(self):
        # Ajouter les nouveaux individus
        if self.temps_global in self.calendrier:
            for id_ind, param in self.calendrier[self.temps_global]:
                seed = self.rng.integers(0, 1_000_000)
                for nom_pol, pol in self.politiques.items():
                    clone = Individu(
                        f"{id_ind}_{nom_pol}",
                        true_param={'alpha': self.true_alpha, **param},
                        sigma=self.true_sigma,
                        seed=seed
                    )
                    clone.politique = pol
                    clone.nom_politique = nom_pol
                    getattr(self, f'individus_actifs_{nom_pol}').append(clone)
                    print(f"[{self.temps_global}] Arrivée de {id_ind} avec politique {nom_pol}")

        sigma = self.true_sigma

        for nom_pol, pol in self.politiques.items():
            alpha_ = getattr(self, f'alpha_{nom_pol}')
            individus_actifs = getattr(self, f'individus_actifs_{nom_pol}')
            hist = getattr(self, f'historique_{nom_pol}')

            alpha, liste_phi = pol.estimation_parametre(alpha_, individus_actifs, hist)
            # Stocker les estimations
            getattr(self, f'estimations_alpha_{nom_pol}').append(alpha)
            getattr(self, f'estimations_phi_{nom_pol}').append(liste_phi)
            nouveaux_actifs = []
            for individu in individus_actifs:

                seed = self.rng.integers(0, 1_000_000)
                action = individu.politique.choisir_action(
                    individu, alpha, phi = liste_phi[individu.id],
                    x_min=self.x_min, x_max=self.x_max,
                    ecart_de_pas=self.ecart_de_pas, cout=self.cout,
                    sigma=sigma, seed=seed
                )
                etat, fini = individu.step(action, cout=self.cout)

                if pol.besoin_historique:
                    hist["action"].append(action)
                    hist["patient"].append(individu.id)
                    hist["XT+1"].append(etat[0])
                    hist["XT"].append(etat[1])
                    hist["XT-1"].append(etat[2])
                    hist["temps"].append(self.temps_global)

                if not fini:
                    nouveaux_actifs.append(individu)
                else:
                    getattr(self, f'individus_termines_{nom_pol}').append(individu)

            setattr(self, f'individus_actifs_{nom_pol}', nouveaux_actifs)

        self.temps_global += 1
        print('temps global', self.temps_global)


    def run(self, t_max):
        for _ in range(t_max):
            self.step_global()


    
    def get_Historique(self):
        """
        Retourne les historiques complets pour chaque politique.
        Format :
        {
            "nom_politique": {
                "XT": [...],
                "XT-1": [...],
                "XT+1": [...],
                "action": [...],
                "patient": [...],
                "temps": [...]
            },
            ...
        }
        """
        resultats = {}
        for nom, politique_instance in self.politiques.items():
            histoo = getattr(self, f'historique_{nom}')
            # On fait une copie pour éviter les effets de bord
            resultats[nom] = {cle: list(val) for cle, val in histoo.items()}
        return resultats
