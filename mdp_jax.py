"""
JAX implementation of MDP value function computation.
Converts the R backend server logic to a fast, jittable JAX function.
"""

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"


import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import io_callback
from jax.scipy import special as jax_special
import time
from typing import Callable, Optional, Tuple
from tqdm import tqdm
import numpy as np


# ============================================================================
# Progress Bar Utilities (from user code)
# ============================================================================


def loop_tqdm(n: int, print_rate: Optional[int] = None, **kwargs) -> Callable:
    """
    Create a tqdm progress bar for a JAX fori_loop.

    Args:
        n (int): Number of iterations.
        print_rate (Optional[int]): Rate at which the progress bar will be updated.
        **kwargs: Extra keyword arguments to pass to tqdm.

    Returns:
        Callable: Progress bar wrapping function.
    """
    _update_progress_bar, close_tqdm = build_tqdm(n, print_rate, **kwargs)

    def _loop_tqdm(func):
        def wrapper_progress_bar(i, val):
            _update_progress_bar(i)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _loop_tqdm


def build_tqdm(
    n: int, print_rate: Optional[int], **kwargs
) -> Tuple[Callable, Callable]:
    """
    Build the tqdm progress bar on the host.

    Args:
        n (int): Number of iterations.
        print_rate (Optional[int]): Rate at which the progress bar will be updated.
        **kwargs: Extra keyword arguments to pass to tqdm.

    Returns:
        Tuple[Callable, Callable]: Update and close functions for the progress bar.
    """
    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)
    show_progress_bar = kwargs.pop("show_progress_bar", True)

    tqdm_bars = {}

    if print_rate is None:
        print_rate = min(int(n / 20), 250) if n > 20 else 1
    else:
        if print_rate < 1 or print_rate > n:
            raise ValueError(f"Invalid print rate: {print_rate}")

    remainder = n % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(n), disable=not show_progress_bar, **kwargs)
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(int(arg))

    def _update_progress_bar(iter_num):
        _ = lax.cond(
            iter_num == 0,
            lambda _: io_callback(_define_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = lax.cond(
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: io_callback(_update_tqdm, print_rate, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = lax.cond(
            iter_num == n - remainder,
            lambda _: io_callback(_update_tqdm, remainder, None, ordered=True),
            lambda _: None,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        _ = lax.cond(
            iter_num == n - 1,
            lambda _: io_callback(_close_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_progress_bar, close_tqdm


# ============================================================================
# MDP Value Function Computation
# ============================================================================


def compute_mu_array(
    x_grid: jax.Array,
    alpha: jax.Array,
    phi: float,
) -> jax.Array:
    """
    Compute expected next states: E[X_{t+1} | X_t, X_{t-1}, action]

    Formula: mu = (1 - alpha) * ((1 + phi) * x_t - phi * x_{t-1}) + beta

    Args:
        x_grid: Discretized state values, shape (n_states,)
        alpha: Action parameters, shape (n_actions,)
        phi: AR(1) coefficient

    Returns:
        mu_array: Expected next states, shape (n_states, n_states, n_actions)
                  mu_array[i, j, a] = E[X_{t+1} | X_t=x_grid[i], X_{t-1}=x_grid[j], action=a]
    """
    n_states = x_grid.shape[0]
    n_actions = alpha.shape[0]

    # Create meshgrid: x_t shape (n_states, 1), x_{t-1} shape (1, n_states)
    x_t = x_grid[:, None]  # shape (n_states, 1)
    x_tm1 = x_grid[None, :]  # shape (1, n_states)

    # Compute for each action
    mu_array = jnp.zeros((n_states, n_states, n_actions))

    for a in range(n_actions):
        # mu = (1 - alpha[a]) * ((1 + phi) * x_t - phi * x_{t-1})
        # Note: beta[a] = 0 in the R code, but formula includes alpha[a] * x_t * phi
        mu = (1 - alpha[a]) * ((1 + phi) * x_t - phi * x_tm1) + alpha[a] * x_t * phi
        mu_array = mu_array.at[:, :, a].set(mu)

    return mu_array


def compute_instant_cost(
    mu_array: jax.Array,
    sigma: jax.Array,
    cost: jax.Array,
) -> jax.Array:
    """
    Compute instantaneous cost: E[|X_{t+1}| + cost_action]

    Where X_{t+1} ~ N(mu, sigma²), the expected absolute value is:
    E[|X_{t+1}|] = sqrt(2/π) * σ * exp(-μ²/2σ²) + μ * (1 - 2*Φ(-μ/σ))

    Args:
        mu_array: Expected next states, shape (n_states, n_states, n_actions)
        sigma: Action noise std, shape (n_actions,)
        cost: Action costs, shape (n_actions,)

    Returns:
        instant_cost: shape (n_states, n_states, n_actions)
    """
    n_states, _, n_actions = mu_array.shape
    instant_cost = jnp.zeros((n_states, n_states, n_actions))

    for a in range(n_actions):
        mu = mu_array[:, :, a]  # shape (n_states, n_states)
        sigma_a = sigma[a]
        cost_a = cost[a]

        # E[|X|] where X ~ N(mu, sigma²)
        abs_expectation = jnp.sqrt(2 / jnp.pi) * sigma_a * jnp.exp(
            -(mu**2) / (2 * sigma_a**2)
        ) + mu * (1 - 2 * jax_special.ndtr(-mu / sigma_a))

        instant_cost = instant_cost.at[:, :, a].set(abs_expectation + cost_a)

    return instant_cost


def compute_value_step_vectorized(
    v_prev: jax.Array,
    mu_array: jax.Array,
    instant_cost: jax.Array,
    x_grid: jax.Array,
    sigma: jax.Array,
    step: float,
    x_min: float,
    x_max: float,
    boundary_penalty: float = 1e8,
) -> jax.Array:
    """
    Compute value function for one backward induction time step.
    Matches R implementation exactly.

    Args:
        v_prev: Value function from next time step, shape (n_states, n_states, n_actions)
                v_prev[k, i, a] = V(X_{t+1}=k, X_t=i, action=a)
        mu_array: Expected next states, shape (n_states, n_states, n_actions)
                  mu_array[i, j, a] = E[X_{t+1} | X_t=i, X_{t-1}=j, action=a]
        instant_cost: Immediate costs, shape (n_states, n_states, n_actions)
        x_grid: State grid, shape (n_states,)
        sigma: Action noise, shape (n_actions,)
        step: Discretization step size
        x_min: Minimum state value (x_grid[0])
        x_max: Maximum state value
        boundary_penalty: Penalty for going out of bounds

    Returns:
        v_current: Value function, shape (n_states, n_states, n_actions)
                   v_current[i, j, a] = V(X_t=i, X_{t-1}=j, action=a)
    """
    n_states = x_grid.shape[0]
    n_actions = sigma.shape[0]

    # Min over actions for each (X_{t+1}, X_t) pair
    # v_prev[k, i, :] contains costs for all actions at (X_{t+1}=k, X_t=i)
    min_over_actions = jnp.min(v_prev, axis=2)  # shape (n_states, n_states)
    # min_over_actions[k, i] = min over actions of V(X_{t+1}=k, X_t=i, *)

    v_current = jnp.zeros((n_states, n_states, n_actions))

    for a in range(n_actions):
        mu_curr = mu_array[:, :, a]  # shape (n_states, n_states)
        # mu_curr[i, j] = E[X_{t+1} | X_t=i, X_{t-1}=j]
        sigma_a = sigma[a]

        # Loop over X_t (current state)
        for i in range(n_states):
            # For fixed X_t=i, get expected next states for all X_{t-1}=j
            mu_for_prev_states = mu_curr[i, :]  # shape (n_states,)
            # mu_for_prev_states[j] = E[X_{t+1} | X_t=i, X_{t-1}=j]

            # Compute probability that X_{t+1} falls in each bin k
            # For each j (prev state), compute P(X_{t+1}=k | X_t=i, X_{t-1}=j)
            # Broadcasting: mu_for_prev_states shape (n_states,) for j, x_grid for k

            # For each j, k pair: P(X_{t+1} in bin k | mu[j], sigma)
            mu_expanded = mu_for_prev_states[:, None]  # shape (n_states_j, 1)
            x_upper = x_grid[None, :] + step / 2  # shape (1, n_states_k)
            x_lower = x_grid[None, :] - step / 2  # shape (1, n_states_k)

            cdf_upper = jax_special.ndtr((x_upper - mu_expanded) / sigma_a)
            cdf_lower = jax_special.ndtr((x_lower - mu_expanded) / sigma_a)
            cdf_diff = cdf_upper - cdf_lower  # shape (n_states_j, n_states_k)
            # cdf_diff[j, k] = P(X_{t+1} in bin k | X_t=i, X_{t-1}=j)

            # Expected future cost per previous state j:
            # For each j: sum_k P(X_{t+1}=k | X_t=i, X_{t-1}=j) * min_over_actions[k, i]
            # min_over_actions[k, i] = min over actions of V(X_{t+1}=k, X_t=i, *)

            # cdf_diff[j, k], min_over_actions[k, i]
            # We need: for each j, sum over k: cdf_diff[j, k] * min_over_actions[k, i]
            future_cost_from_model = jnp.dot(
                cdf_diff, min_over_actions[:, i]
            )  # shape (n_states,)

            # Boundary penalties - MATCH R IMPLEMENTATION EXACTLY
            # R uses x_values[1] for BOTH boundaries (appears to be a bug, but we match it)
            x_first = x_grid[0]  # This is x_min
            left_penalty = (
                jax_special.ndtr((x_first - step / 2 - mu_for_prev_states) / sigma_a)
                * boundary_penalty
            )
            # R formula: (1 - pnorm(-x_values[1] + step/2, ...))
            # = (1 - pnorm(-(x_min) + step/2, ...))
            right_penalty = (
                1
                - jax_special.ndtr((-x_first + step / 2 - mu_for_prev_states) / sigma_a)
            ) * boundary_penalty

            future_cost_total = future_cost_from_model + left_penalty + right_penalty

            # V(X_t=i, X_{t-1}=j, action=a)
            v_current = v_current.at[i, :, a].set(
                instant_cost[i, :, a] + future_cost_total
            )

    return v_current


def compute_value_step_jitted(
    V_prev: jax.Array,
    mu_array: jax.Array,
    instant_cost: jax.Array,
    x_grid: jax.Array,
    sigma: jax.Array,
    step: float,
    x_min: float,
    x_max: float,
    boundary_penalty: float = 1e8,
) -> jax.Array:
    """
    Fully JIT-compiled version of compute_value_step_vectorized.
    Uses vmap and pure JAX operations. Matches R implementation exactly.
    """
    n_states = x_grid.shape[0]
    n_actions = sigma.shape[0]

    min_over_actions = jnp.min(V_prev, axis=2)  # shape (n_states, n_states)
    # min_over_actions[k, i] = min over actions of V(X_{t+1}=k, X_t=i, *)

    def compute_for_action_current_state(state_action_idx):
        """Compute V for a single (action, current_state) pair."""
        a = state_action_idx // n_states
        i = state_action_idx % n_states

        mu_for_prev_states = mu_array[i, :, a]  # shape (n_states,)
        # mu_for_prev_states[j] = E[X_{t+1} | X_t=i, X_{t-1}=j]
        sigma_a = sigma[a]

        # CDF computation
        # For each prev state j and future state k, compute:
        # P(X_{t+1} in bin k | X_t=i, X_{t-1}=j)
        x_upper = x_grid[None, :] + step / 2
        x_lower = x_grid[None, :] - step / 2
        mu_expanded = mu_for_prev_states[:, None]

        cdf_upper = jax_special.ndtr((x_upper - mu_expanded) / sigma_a)
        cdf_lower = jax_special.ndtr((x_lower - mu_expanded) / sigma_a)
        cdf_diff = cdf_upper - cdf_lower  # shape (n_states_j, n_states_k)
        # cdf_diff[j, k] = P(X_{t+1} in bin k | X_t=i, X_{t-1}=j)

        # Future cost:
        # For each prev state j: sum_k P(X_{t+1}=k | X_t=i, X_{t-1}=j) * min_over_actions[k, i]
        # cdf_diff[j, k], min_over_actions[k, i]
        # We need: for each j, sum over k: cdf_diff[j, k] * min_over_actions[k, i]
        future_cost_from_model = jnp.dot(
            cdf_diff, min_over_actions[:, i]
        )  # shape (n_states,)

        # Boundary penalties - MATCH R IMPLEMENTATION EXACTLY
        x_first = x_grid[0]  # This is x_min
        left_penalty = (
            jax_special.ndtr((x_first - step / 2 - mu_for_prev_states) / sigma_a)
            * boundary_penalty
        )
        right_penalty = (
            1 - jax_special.ndtr((-x_first + step / 2 - mu_for_prev_states) / sigma_a)
        ) * boundary_penalty

        future_cost_total = future_cost_from_model + left_penalty + right_penalty
        v_result = instant_cost[i, :, a] + future_cost_total

        return v_result

    # Vectorize over all (action, current_state) pairs
    v_flat = jax.vmap(compute_for_action_current_state)(
        jnp.arange(n_actions * n_states)
    )

    # Reshape back to (n_states, n_states, n_actions)
    # v_flat has shape (n_actions * n_states, n_states) where each row is for (a, i)
    # We need to reshape to (n_states, n_states, n_actions)
    # The indexing in v_flat is: v_flat[a*n_states + i, j]
    # We want: v_current[i, j, a]
    v_current = jnp.zeros((n_states, n_states, n_actions))
    for idx in range(n_actions * n_states):
        a = idx // n_states
        i = idx % n_states
        v_current = v_current.at[i, :, a].set(v_flat[idx, :])

    return v_current


def _backward_induction_final_only(
    x_grid: jax.Array,
    alpha: jax.Array,
    sigma: jax.Array,
    cost: jax.Array,
    phi: float,
    step: float,
    x_min: float,
    x_max: float,
    T: int = 120,
) -> jax.Array:
    """
    JIT-compiled backward induction returning only final timestep.

    Returns:
        V: shape (n_states, n_states, n_actions)
    """
    # Pre-compute mu_array and instant_cost
    mu_array = compute_mu_array(x_grid, alpha, phi)
    instant_cost = compute_instant_cost(mu_array, sigma, cost)

    # Initialize: V at final time step = instant cost
    V_current = instant_cost  # shape (n_states, n_states, n_actions)

    # Wrapped step function for fori_loop
    def step_func(t, V_prev):
        V_next = compute_value_step_jitted(
            V_prev,
            mu_array,
            instant_cost,
            x_grid,
            sigma,
            step,
            x_min,
            x_max,
        )
        return V_next

    # Run backward induction for T-1 steps
    V_final = lax.fori_loop(0, T - 1, step_func, V_current)
    return V_final


# JIT-compile the final-only version
backward_induction_final_only_jitted = jax.jit(_backward_induction_final_only, static_argnums=(7, 8))


def _backward_induction_all_timesteps(
    x_grid: jax.Array,
    alpha: jax.Array,
    sigma: jax.Array,
    cost: jax.Array,
    phi: float,
    step: float,
    x_min: float,
    x_max: float,
    T: int = 120,
) -> jax.Array:
    """
    JIT-compiled backward induction storing all timesteps.

    Returns:
        V: shape (T, n_states, n_states, n_actions)
    """
    # Pre-compute mu_array and instant_cost
    mu_array = compute_mu_array(x_grid, alpha, phi)
    instant_cost = compute_instant_cost(mu_array, sigma, cost)

    # Initialize: V at final time step = instant cost
    V_current = instant_cost  # shape (n_states, n_states, n_actions)

    n_states = x_grid.shape[0]
    n_actions = alpha.shape[0]

    # Initialize storage array for all timesteps
    V_all = jnp.zeros((T, n_states, n_states, n_actions))
    V_all = V_all.at[T - 1, :, :, :].set(V_current)

    # Wrapped step function that stores results
    def step_func_store(t, state):
        V_prev, V_all_prev = state
        V_next = compute_value_step_jitted(
            V_prev,
            mu_array,
            instant_cost,
            x_grid,
            sigma,
            step,
            x_min,
            x_max,
        )
        # Store at time T - 2 - t (backward from T-1 to 0)
        V_all_next = V_all_prev.at[T - 2 - t, :, :, :].set(V_next)
        return V_next, V_all_next

    # Run backward induction and store all timesteps
    V_final, V_all_final = lax.fori_loop(0, T - 1, step_func_store, (V_current, V_all))
    return V_all_final


# JIT-compile the all-timesteps version
backward_induction_all_timesteps_jitted = jax.jit(_backward_induction_all_timesteps, static_argnums=(7, 8))


def backward_induction(
    alpha: jax.Array,
    sigma: jax.Array,
    cost: jax.Array,
    phi: float,
    x_min: float,
    x_max: float,
    step: float,
    T: int = 120,
    show_progress: bool = True,
    return_all_timesteps: bool = False,
) -> jax.Array:
    """
    Compute optimal value function via backward induction.

    This solves the MDP by working backward from the final time step.
    Matches R server output format exactly.

    Args:
        alpha: Action parameters, shape (n_actions,)
        sigma: Action noise std, shape (n_actions,)
        cost: Action costs, shape (n_actions,)
        phi: AR(1) coefficient
        x_min: Minimum state value
        x_max: Maximum state value
        step: Discretization step size
        T: Time horizon (number of steps)
        show_progress: Whether to show progress bar
        return_all_timesteps: If True, return all timesteps. If False (default),
                              return final timestep only to match R server behavior.

    Returns:
        V: Value function
           If return_all_timesteps=False: shape (n_actions, 1, n_states, n_states)
              V[action, time, x_curr_idx, x_prev_idx] = expected cumulative cost at final time
           If return_all_timesteps=True: shape (n_actions, T, n_states, n_states)
              V[action, time, x_curr_idx, x_prev_idx] = expected cumulative cost at each time
           Indexing matches R: V[action, time, x_current, x_previous]
    """
    # Create state grid (before JIT)
    x_grid = jnp.arange(x_min, x_max + step / 2, step)
    n_states = x_grid.shape[0]
    n_actions = alpha.shape[0]

    #print(f"Computing MDP value function...")
    #print(f"  State space: {n_states} states in [{x_min}, {x_max}] with step {step}")
    #print(f"  Actions: {n_actions}")
    #print(f"  Time horizon: {T}")
    #print(f"  Total array size: {n_actions * T * n_states * n_states:,} elements")

    # Run JIT-compiled backward induction
    #print(f"Backward induction (JIT-compiled): {T-1} steps")

    start_time = time.time()
    if return_all_timesteps:
        V_result = backward_induction_all_timesteps_jitted(
            x_grid, alpha, sigma, cost, phi, step, x_min, x_max, T
        )
    else:
        V_result = backward_induction_final_only_jitted(
            x_grid, alpha, sigma, cost, phi, step, x_min, x_max, T
        )
    jit_time = time.time() - start_time

    #print(f"JIT compilation + execution: {jit_time:.2f}s")

    if return_all_timesteps:
        # V_result is (T, n_states, n_states, n_actions)
        # Need to transpose to (n_actions, T, n_states, n_states)
        V = jnp.transpose(V_result, (3, 0, 1, 2))  # (n_actions, T, n_states, n_states)
    else:
        # V_result is (n_states, n_states, n_actions) = (x_curr, x_prev, action)
        # Transpose to (n_actions, n_states, n_states) then add time dimension
        V_transposed = jnp.transpose(V_result, (2, 0, 1))  # (n_actions, n_states, n_states)
        V = V_transposed[:, None, :, :]  # (n_actions, 1, n_states, n_states)

    return V


# ============================================================================
# Demonstration and Benchmarking
# ============================================================================


def demonstrate_mdp():
    """Demonstrate the MDP computation with default parameters."""

    # Use parameters similar to the notebook
    alpha = jnp.array([0.0, 1.0])
    sigma = jnp.array([0.1, 0.1])
    cost = jnp.array([0.0, 50.0])
    phi = 0.5

    x_min = -20.0
    x_max = 20.0
    step = 0.05
    T = 120

    # Compute value function (final timestep only)
    print("\n" + "=" * 70)
    print("DEMONSTRATION: MDP Value Function Computation")
    print("=" * 70)

    print("\n--- Mode 1: Final timestep only (default) ---")
    start_time = time.time()
    V = backward_induction(
        alpha, sigma, cost, phi, x_min, x_max, step, T, show_progress=True
    )
    elapsed = time.time() - start_time

    print(f"\nComputation completed in {elapsed:.2f}s")
    print(f"Output shape: {V.shape}")
    print(f"  Dimension 0 (actions): {V.shape[0]}")
    print(f"  Dimension 1 (time): {V.shape[1]}")
    print(f"  Dimension 2 (current state): {V.shape[2]}")
    print(f"  Dimension 3 (previous state): {V.shape[3]}")

    # Show some example values
    n_states = V.shape[2]
    mid = n_states // 2
    print("\n--- Example Values (Final Timestep) ---")
    print(f"V[action=0, time=0, x_curr=mid, x_prev=mid] = {V[0, 0, mid, mid]:.4f}")
    print(f"V[action=1, time=0, x_curr=mid, x_prev=mid] = {V[1, 0, mid, mid]:.4f}")

    # Show value function statistics
    print("\n--- Value Function Statistics ---")
    print(f"Min value: {jnp.min(V):.4f}")
    print(f"Max value: {jnp.max(V):.4f}")
    print(f"Mean value: {jnp.mean(V):.4f}")
    print(f"Std dev: {jnp.std(V):.4f}")

    # Demonstrate agent decision-making
    print("\n--- Example: Agent Decision Making ---")
    print(f"If an agent is at state (0, 0) [index {mid}]:")
    time_idx = 0

    cost_action_0 = V[0, time_idx, mid, mid]
    cost_action_1 = V[1, time_idx, mid, mid]

    print(f"  Cost with action 0: {cost_action_0:.4f}")
    print(f"  Cost with action 1: {cost_action_1:.4f}")

    if cost_action_0 < cost_action_1:
        print(f"  → Optimal choice: Action 0")
    else:
        print(f"  → Optimal choice: Action 1")

    # Demonstrate full timestep output
    print("\n--- Mode 2: All timesteps ---")
    start_time = time.time()
    V_all = backward_induction(
        alpha, sigma, cost, phi, x_min, x_max, step, T,
        show_progress=False, return_all_timesteps=True
    )
    elapsed = time.time() - start_time

    print(f"\nComputation completed in {elapsed:.2f}s")
    print(f"Output shape: {V_all.shape}")
    print(f"  Dimension 0 (actions): {V_all.shape[0]}")
    print(f"  Dimension 1 (time): {V_all.shape[1]}")
    print(f"  Dimension 2 (current state): {V_all.shape[2]}")
    print(f"  Dimension 3 (previous state): {V_all.shape[3]}")

    print("\n--- Example Values across Timesteps ---")
    for t in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        print(f"T={t:3d}: V[0, {t}, mid, mid] = {V_all[0, t, mid, mid]:10.4f}, " +
              f"V[1, {t}, mid, mid] = {V_all[1, t, mid, mid]:10.4f}")

    print("\n--- Timestep Evolution ---")
    print(f"Time evolution of V[0, :, mid, mid] (action 0):")
    v0_evolution = V_all[0, :, mid, mid]
    print(f"  Min: {jnp.min(v0_evolution):.4f}, Max: {jnp.max(v0_evolution):.4f}")
    print(f"  First timestep (t=0): {v0_evolution[0]:.4f}")
    print(f"  Last timestep (t={T-1}): {v0_evolution[T-1]:.4f}")

    return V, V_all


def benchmark_mdp():
    """Benchmark MDP computation with different parameter sizes."""

    print("\n" + "=" * 70)
    print("BENCHMARKING: MDP Value Function Computation")
    print("=" * 70)

    # Test different discretization steps
    test_cases = [
        {"step": 0.2, "name": "Coarse (step=0.2)"},
        {"step": 0.1, "name": "Medium (step=0.1)"},
        {"step": 0.05, "name": "Fine (step=0.05)"},
    ]

    alpha = jnp.array([0.0, 0.8])
    sigma = jnp.array([0.5, 0.5])
    cost = jnp.array([0.0, 201.0])
    phi = 0.5
    x_min = -20.0
    x_max = 20.0
    T = 120

    results = []

    for test_case in test_cases:
        step = test_case["step"]
        name = test_case["name"]

        x_grid = jnp.arange(x_min, x_max + step / 2, step)
        n_states = x_grid.shape[0]
        n_elements = 2 * T * n_states * n_states

        print(f"\n{name}")
        print(f"  n_states: {n_states}")
        print(f"  Array size: {n_elements:,} elements ({n_elements * 8 / 1e6:.1f} MB)")

        start_time = time.time()
        V = backward_induction(
            alpha, sigma, cost, phi, x_min, x_max, step, T, show_progress=False
        )
        elapsed = time.time() - start_time

        per_step = elapsed / (T - 1)

        print(f"  Total time: {elapsed:.3f}s")
        print(f"  Time per step: {per_step*1000:.2f}ms")
        print(f"  Throughput: {n_elements/1e6 / elapsed:.1f} M elements/s")

        results.append(
            {
                "name": name,
                "step": step,
                "n_states": n_states,
                "time": elapsed,
                "per_step": per_step,
            }
        )

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Time (s)':<12} {'Per Step (ms)':<15}")
    print("-" * 52)
    for result in results:
        print(
            f"{result['name']:<25} {result['time']:<12.3f} {result['per_step']*1000:<15.2f}"
        )


if __name__ == "__main__":
    # Run demonstration
    V, V_all = demonstrate_mdp()

    # Run benchmarks
    benchmark_mdp()

    print("\n" + "=" * 70)
    print("JAX MDP Value Function - Computation Complete")
    print("=" * 70)
