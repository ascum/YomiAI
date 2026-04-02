import numpy as np

# Simulate a "Personality": A fixed preference vector
# In a real system, this would be the hidden taste of a specific user.
_USER_W = None

def reset_user():
    global _USER_W
    _USER_W = None

def get_reward(item_id, item_vec):
    global _USER_W
    if _USER_W is None:
        # Initialize the personality once
        _USER_W = np.random.randn(len(item_vec))

    # Calculate match probability using the sigmoid function
    # w @ item_vec is the dot product (similarity)
    p = 1 / (1 + np.exp(-_USER_W @ item_vec))
    
    
    return int(np.random.rand() < p)
