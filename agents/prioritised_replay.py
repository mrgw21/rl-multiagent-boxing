import numpy as np


class PrioritisedReplay:
    """
    Class for prioritised experience replay.
    Handles adding, updating and sampling from an ordered set of experiences.
    Experiences are ordered based on TD error. 
    """
    def __init__(self, max_len=1000):
        self.max_len = max_len
        #Replay buffer is a list of dictionaries. Each dictionary stores the experiences and its td error
        # e.g. {experience: [state, action, next_reward, next_state], td_error: td_error_val}
        self.replay_buffer = []

    def sort_replay_buffer(self):
        """Sorts the replay buffer in reverse order for TD Error key"""
        self.replay_buffer.sort(key=lambda x: x["TD Error"], reverse=True)

    def add_experience(self, experience, td_error):
        """
        Adds an experience to the replay, placing it in a location based on its td error
        """
        # If the buffer is longer than the maximum length, only add it 
        dict_entry = {"Experience": experience, "TD Error": td_error}
        # If buffer is already at maximum capacity, remove last element (lowest priority)
        if len(self.replay_buffer) == self.max_len:
            self.replay_buffer.pop()

        # Add new experience entry and resort buffer
        self.replay_buffer.append(dict_entry)
        self.sort_replay_buffer()
    
    def get_sample(self, num_samples=32):
        """Samples a batch from the buffer, increasing probability of choosing those with higher TD Errors"""
        # If there are no samples available, return nothing 
        if len(self.replay_buffer) == 0:
            return []
        
        # Get every TD Error from the batch and then make it into a numpy array
        td_errors = []
        for entry in self.replay_buffer:
            td_errors.append(entry["TD Error"])
        td_errors = np.array(td_errors)
        
        # Calculate probabilites - Add a tiny number to prevent division by zero
        probabilities = (td_errors + 1e-6) / (td_errors.sum() + 1e-5)

        # Using an array of values and an array of probabilities, can get 
        # n samples (where n = num_samples) from the replay
        chosen_indices = np.random.choice(len(self.replay_buffer), size=num_samples, p=probabilities)
        chosen_samples = [self.replay_buffer[i] for i in chosen_indices]
        return chosen_samples


    
    def update_existing_entry(self, experience, new_td_error):
        """Updates the TD Error of an existing experience"""
        for dict_entry in self.replay_buffer:
            # array_equal prevents numpy error 
            if np.array_equal(dict_entry["Experience"], experience):
                dict_entry["TD Error"] = new_td_error
                self.sort_replay_buffer()
                return
        self.add_experience(experience, new_td_error)


