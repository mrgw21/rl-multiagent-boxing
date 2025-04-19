import numpy as np

class ReplayMemory():
    def __init__(self,maxLen):
        self.replay_buffer = []
        self.max_capacity = maxLen
        self.replay_buffer = []
        self.priorities = []
        self.batch_size = 32
        # "how much prioritisation is used" - when exp_alpha is 0, all samples have the same probability of being sampled
        self.exp_alpha = 0.6
        # Importance sampling exponent - prevents bias of sampling from only high priority samples
        self.exp_beta = 0.4 
    
    def store_experience(self, experience, priority, cache=True):
        """
        Stores an experience + its priority in the relevant lists
        """
        # If we are over capacity, remove the lowest priority experience + priority (last one)
        if len(self.replay_buffer) >= self.max_capacity:
            self.replay_buffer.pop(0)
            self.priorities.pop(0)
        # Add new experience and priority
        self.replay_buffer.append(experience)
        self.priorities.append(priority)

        if cache:
            # Refresh cache every self.refresh_frequency steps
            self.steps_since_last_update += 1
            if self.steps_since_last_update >= self.refresh_frequency:
                self.update_cache() # build_cache function from Daley & Amato
                self.steps_since_last_update = 0
        
    def update_cache(self):
        """ 
        Implementatino of the build_cache function from Daley & Amato - refreshes the cache
        """
        self.cache = [] # Initialize empty list C
        # for 1, 2, . . . , S/B do
        for i in range(self.cache_size // self.block_size):
            # Sample block (sk, ak, rk, sk+1), . . . , (sk+B−1, ak+B−1, rk+B−1, sk+B ) from D
            if len(self.replay_buffer) <= self.block_size:
                sample_block = self.replay_buffer.copy()
            else:
                sk = np.random.randint(0, len(self.replay_buffer) - self.block_size)
                sample_block = self.replay_buffer[sk:sk + self.block_size]
            
            sample_block = self.replay_buffer[sk : sk+self.block_size]

            # Rλ ← max(a′ ∈ A) Q(sk+B,a′;θ)
            lam_returns = []
            next_return = 0

            #for i ∈ {k + B − 1, k + B − 2, . . . , k} do
            for sample in reversed(sample_block):
                state, action, reward, next_state, next_action, finished = sample

                # Rλ ← ri
                if finished:
                    lam_return = reward
                # Rλ ← ri + γ[λRλ + (1 − λ) maxa′ ∈A Q(ˆsi+1, a′; θ)]
                else:
                    q_val_1 = self.value(next_state, next_action, self.weights_1) 
                    q_val_2 = self.value(next_state, next_action, self.weights_2) 
                    q_estimate = (q_val_1 + q_val_2) / 2
                    lam_return = reward + self.gamma * (self.lam_val * next_return + (1-self.lam_val) * q_estimate)
                #Append tuple (ˆsi, ai, Rλ) to C
                lam_returns.append((state, action, lam_return))
                next_return = lam_return

            self.cache.extend(reversed(lam_returns))

    
    def prioritised_sample(self):
        """
        Sample a set of experiences based on their value/priority - determined using td_error
        """
        if not self.replay_buffer:
            return [], [], []
        
        # Create a copy array for the priorities 
        priority_arr = np.array(self.priorities, dtype=np.float64)

        # Additional check to make sure no values are nan in the prioity list or the total sum = 0
        if np.sum(priority_arr) == 0 or np.any(np.isnan(priority_arr)):
            priority_arr = np.ones_like(priority_arr) # Sets all priorities to 1 so probability of selecting experience is equal

        # Create probability array using alpha priority value -  determines probability of choosing action from the priority values (Schaul et al., 2015)
        probs = priority_arr ** self.exp_alpha
        probs /= np.sum(probs)  # Divide the array by its own sum to ensure it sums up to 1 (needed to for a prbability distribution)
        
        # To ensure sample size doesn't exceed number of non-zero probabilities
        num_non_zeros = np.count_nonzero(probs)
        # Calculate possible sample size by taking min of possible values
        sample_size = min(self.batch_size, len(self.replay_buffer), num_non_zeros)

        # If sample size is 0, use uniform sampling again
        if sample_size == 0:
            probs = np.ones(len(priority_arr)) / len(priority_arr)
            sample_size = min(self.batch_size, len(self.replay_buffer))

        # An extra check to prevent nan probabilities
        if np.any(np.isnan(probs)):
            probs = np.ones(len(priority_arr)) / len(priority_arr) # Make probabilities all the same in the array 
            sample_size = min(self.batch_size, len(self.replay_buffer))

        # Can reuse samples if sample size is greater than the number of non zero samples
        if sample_size > num_non_zeros:
            reuse_samples = True
        else:
            reuse_samples = False
        # Randomly choose the indices of the experiences using these probabilities - replace prevents sampling the same experience twice
        idx_vals = np.random.choice(len(self.replay_buffer), min(self.batch_size, 
                                len(self.replay_buffer)), p=probs, replace=reuse_samples)
        
        # Importance sampling weights to stop the agent only choosing high priority values - from Schaul et al. (2015)
        weights = (len(self.replay_buffer) * probs[idx_vals]) ** (-self.exp_beta)
        weights = weights / np.max(weights)  # Same normalisations as before
        
        # Use calculated indices to extract experiences from self.replay_buffer
        sample_batch = [self.replay_buffer[i] for i in idx_vals]
        return sample_batch, idx_vals, weights
    
    def update_priority_order(self, indices, errors):
        """
        Update the priority list with the latest TD errors
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + 1e-6  # Add a small constant to the absolute TD error to make it more stable
    
    def update_weights(self, action, state, td_error, update_weights):
        """Performs update directly to agent's weight vectors rather than to local weights"""
        if update_weights is self.weights_1:
            self.weights_1[action] += self.alpha * td_error * state
        else:
            self.weights_2[action] += self.alpha * td_error * state