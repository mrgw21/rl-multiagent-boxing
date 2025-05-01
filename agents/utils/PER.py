# Code for prioritised experience replay
import numpy as np


def store_experience(agent, experience, priority, cache=True):
    """
    Stores an experience + its priority in the relevant lists
    """

    # Add new experience and priority
    agent.replay_buffer.append(experience)
    initial_priority = max(abs(priority), 1e-4)
    agent.priorities.append(initial_priority)

    if cache:
        # Refresh cache every agent.refresh_frequency steps
        agent.steps_since_last_update += 1
        if agent.steps_since_last_update >= agent.refresh_frequency:
            update_cache(agent) # build_cache function from Daley & Amato
            agent.steps_since_last_update = 0
        
def update_cache(agent):
    """ 
    Implementation of the build_cache function from Daley & Amato - refreshes the cache
    """
    agent.cache = [] # Initialize empty list C
    replay_buffer_list = list(agent.replay_buffer)  # Convert deque to list
    # for 1, 2, . . . , S/B do
    for i in range(agent.cache_size // agent.block_size):
        # Sample block (sk, ak, rk, sk+1), . . . , (sk+B−1, ak+B−1, rk+B−1, sk+B ) from D
        if len(replay_buffer_list) <= agent.block_size:
            sample_block = replay_buffer_list.copy()
        else:
            sk = np.random.randint(0, len(replay_buffer_list) - agent.block_size)
            sample_block = replay_buffer_list[sk:sk + agent.block_size]

        # Rλ ← max(a′ ∈ A) Q(sk+B,a′;θ)
        lam_returns = []
        next_return = 0

        #for i ∈ {k + B − 1, k + B − 2, . . . , k} do
        for sample in reversed(sample_block):
            state, action, reward, next_state, next_action, finished = sample
            # Rλ ← ri
            lam_return = reward
            # Rλ ← ri + γ[λRλ + (1 − λ) maxa′ ∈A Q(ˆsi+1, a′; θ)]
            if not finished:
                next_q_vals = []
                for poss_action in agent.action_list:
                    q_val_1 = agent.value(next_state, poss_action, agent.target_weights_1) 
                    q_val_2 = agent.value(next_state, poss_action, agent.target_weights_2) 
                    q_val = (q_val_1 + q_val_2) / 2
                    next_q_vals.append(q_val)
                max_q_val = max(next_q_vals)
                lam_return = reward + agent.gamma * (agent.lam_val * next_return + (1-agent.lam_val) * max_q_val)
            #Append tuple (ˆsi, ai, Rλ) to C
            lam_returns.append((state, action, lam_return))
            next_return = lam_return

        agent.cache.extend(reversed(lam_returns))

def prioritised_sample(agent):
    """
    Sample a set of experiences based on their value/priority - determined using td_error
    """
    if not agent.replay_buffer:
        return [], [], []
    
    # Create a copy array for the priorities 
    priority_arr = np.array(agent.priorities, dtype=np.float64)

    # Additional check to make sure no values are nan in the prioity list or the total sum = 0
    if np.sum(priority_arr) == 0 or np.any(np.isnan(priority_arr)):
        probs = np.ones(len(priority_arr)) / len(priority_arr)
        # priority_arr = np.ones_like(priority_arr) # Sets all priorities to 1 so probability of selecting experience is equal
    else:
        # Create probability array using alpha priority value -  determines probability of choosing action from the priority values (Schaul et al., 2015)
        probs = priority_arr ** agent.exp_alpha
        probs /= np.sum(probs)  # Divide the array by its own sum to ensure it sums up to 1 (needed to for a prbability distribution)
    
    # Calculate possible sample size 
    current_buffer_length = len(agent.replay_buffer)
    sample_size = min(agent.batch_size, current_buffer_length)

    # An extra check to prevent nan probabilities
    if not np.isclose(np.sum(probs), 1.0):
        probs = np.ones(current_buffer_length) / current_buffer_length # Make probabilities all the same in the array 

    # Randomly choose the indices of the experiences using these probabilities - replace prevents sampling the same experience twice
    idx_vals = np.random.choice(current_buffer_length, sample_size, p=probs, replace=True)
    
    # Importance sampling weights to stop the agent only choosing high priority values - from Schaul et al. (2015)
    weights = (current_buffer_length * probs[idx_vals]) ** (-agent.exp_beta)
    
    if np.max(weights) > 0:
        weights /= np.max(weights)  # Same normalisations as before
    else:
        weights = np.ones_like(weights)
    
    # Use calculated indices to extract experiences from agent.replay_buffer
    sample_batch = [agent.replay_buffer[i] for i in idx_vals]
    return sample_batch, idx_vals, weights

def update_priority_order(agent, indices, errors):
    """
    Update the priority list with the latest TD errors
    """
    for idx, error in zip(indices, errors):
        agent.priorities[idx] = max(abs(error), 1e-6)  # Ensure at least a small value to make it more stable