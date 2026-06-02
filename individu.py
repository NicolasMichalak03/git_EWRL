import numpy as np

class Individu:
    def __init__(self, id_individu, parametre_cache , sigma=0.1, state_initial=0.0, max_steps=119,seed=None):
        """
        Represents a single individual in the simulation.

        Parameters
        ----------
        id_individu : str
            Unique identifier for the individual.
        parametre_cache : dict
            Hidden parameters (true alpha, true phi).
        sigma : float
            Standard deviation of the noise in state transitions.
        state_initial : float
            Initial state value.
        max_steps : int
            Maximum number of time steps before termination.
        seed : int
            Random seed for reproducibility.
        """
        self.id = id_individu
        self.true_param = parametre_cache
        self.sigma = sigma
        self.state_initial = state_initial
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.default_rng(seed) 
        self.reset()
    def reset(self):
        """Resets the individual's trajectory and internal state."""
        self.historique_rewards = []
        self.historique_states = [self.state_initial]
        self.historique_actions = []
        self.state_prev = self.state_initial
        self.state_curr = self.state_initial
        self.step_count = 0
        self.done = False

    def step(self, action,cout):
        """
        Applies one time-step transition given an action.

        The state evolves according to:
            new_state = (1 - alpha) * ((1 + phi) * x_t - phi * x_{t-1})
                        + alpha * (phi * x_t)
                        + noise

        Parameters
        ----------
        action : int
            Chosen action.
        cout : list or array
            Cost associated with each action.

        Returns
        -------
        (state_curr, state_prev, state_prev_prev), done
        """
        if self.done:
            print("erreur ? Je crois qu'on doit pas avoir de pas si il est déjà terminé")
            return self.state_curr, True

        self.historique_actions.append(action)

        alpha = self.true_param['alpha'][action]
        phi = self.true_param['phi']
        bruit = self.rng.normal(0, self.sigma)

        new_state = (
                        (1-alpha) * ((1 + phi) * self.state_curr - phi * self.state_prev) 
                      + alpha*(phi * self.state_curr) + bruit
        )
        
        # Status update
        state_prev_temp =  self.state_prev
        self.state_prev = self.state_curr
        self.state_curr = new_state
        self.historique_states.append(new_state)
        
        reward = abs(new_state) + cout[action]
        self.historique_rewards.append(reward)

        # estimates
            #Must do nothing if we estimate the phi parameters from a global point of view
        self.estimer_parametre()

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        return [self.state_curr , self.state_prev, state_prev_temp], self.done


    def estimer_parametre(self):
        return
    
    def is_done(self):
        return self.done

    def get_state(self):
        return self.state_curr

    def get_historique_actions(self):
        return self.historique_actions
    
    def get_historique(self):
        return self.historique_states
    
    def get_historique_rewards(self):
        return self.historique_rewards

