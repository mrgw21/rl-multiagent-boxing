import numpy as np


class LSH:
        def __init__(self):
            
            # Constants. M (hash table size), n (screen bit vector size) l (number of random bit vectors), k (number of non-zero entries)
            # All obtained from Bellemare et al. (2013)
            self.hash_table_size = 50
            self.screen_height = 210
            self.screen_width = 160
            self.screen_bit_vec_size = self.screen_width * self.screen_height * 128 # 128 colours
            self.num_rand_bit_vecs = 2000
            self.num_non_zero_entries = 1000
            
            # Initialisation (once).
            # {v1 . . . vl} ← generateRandomVectors(l, k, n)
            # {hash1 . . . hashl} ← generateHashFunctions(l, M, n)
            self.vecs = self.generate_random_vectors(self.num_rand_bit_vecs, self.num_non_zero_entries, self.screen_bit_vec_size)
            self.hash_funcs = self.generate_hash_funcs(self.num_rand_bit_vecs, self.hash_table_size, self.screen_bit_vec_size)
            
        
        def generate_hash_funcs(self, num_rand_bit_vecs, hash_table_size, screen_bit_vec_size):
            # Initialise hash1 . . . hashl ∈ Rn = 0
            hash_funcs = np.zeros((num_rand_bit_vecs, screen_bit_vec_size), dtype=np.int8)
            # for i = 1 . . . l, j = 1 . . . n do
            for i in range(num_rand_bit_vecs):
                for j in range(num_rand_bit_vecs):
                    # hashi[j] ← random(1, M ) (uniformly random coordinate between 1 and M)
                    hash_funcs[i][j] = np.random.randint(0, hash_table_size)
            return hash_funcs
             
        
        def binarise_screen(self, screen):
            # Initialise s ∈ Rn = 0
            s = np.zeros((self.screen_bit_vec_size), dtype=np.int8)
            
            # for y = 1 . . . h, x = 1 . . . w (h = 210, w = 160) do
            for y in range(min(self.screen_height, screen.shape[0])):
                for x in range(min(self.screen_width, screen.shape[1])):
                    # For RGB images, convert to a single intensity value
                    # Can use average of RGB or grayscale conversion
                    if len(screen.shape) == 3:  # RGB image
                        # Use average of RGB values as intensity (simple grayscale conversion)
                        intensity_val = np.mean(screen[y, x]).astype(np.int32)
                    else:  # Already grayscale
                        intensity_val = int(screen[y, x])
                    
                    # Make sure intensity_val is within bounds for our color range
                    intensity_val = min(127, int(intensity_val))
                    
                    # Calculate index safely with proper scaling
                    index = np.int64(x + (y * self.screen_width) + (intensity_val * self.screen_width * self.screen_height))
                    
                    # Add a safeguard to prevent index out of bounds
                    if 0 <= index < self.screen_bit_vec_size:
                        s[index] = 1
            return s
            
        
        def generate_random_vectors(self, num_random_bit_vectors, num_non_zero_entries, bit_vector_size):
            # Initialize v1 . . . vl ∈ Rn = 0
            v = np.zeros((num_random_bit_vectors, bit_vector_size), dtype=np.int8)
            for i in range(num_random_bit_vectors):
                # Select x1, x2, . . . , xk distinct coordinates between 1 and n uniformly at random
                xs = np.random.choice(bit_vector_size, num_non_zero_entries, replace=False)
                # vi[x1] = 1; vi[x2] = 1; . . . ; vi[xk] = 1
                v[i, xs] = 1
            return v
    
        def feature_extraction_lsh(self, screen):
            """
            Function that generates Locally Sensitive Hashing (LSH) features from boxing Atari
            Adapted from the pseudocode in Bellemare et al. (2013)
            """
            # s ← binarizeScreen(I) (s has length n)
            s = self.binarise_screen(screen)
            # Initialize φ ∈ RlM = 0
            phi = np.zeros((self.num_rand_bit_vecs * self.hash_table_size), dtype=np.int8)
            # for i = 1 . . . l do
            for i in range(self.num_rand_bit_vecs):
                # h = 0
                h = 0
                # for j = 1 . . . n do
                for j in range(self.screen_bit_vec_size):
                    # h ← h + I[sj =vij ]hashi[j] mod M (hash the projection of s onto vi)
                    if self.vecs[i][j] == 1 and s[j] == 1: # only want when both are 1, not only equal to each other
                        h = (h + self.hash_funcs[i][j]) % self.hash_table_size
                # end for
                # φ[M (i − 1) + h] = 1 (one binary feature per random bit vector)
                phi[i * self.hash_table_size + h] = 1
            # end for