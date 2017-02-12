import numpy as np

# Function to project input matrix X of dimensions mxn 
# to a matrix of dimension mxd, where each element is 
# sampled independently from a univariate normal
# distribution

def rand_proj(X, d):
    # Init
    G = []
    rows = len(X[0])
    cols = d
    mean = 0.0
    sd = 1
    
    # Calculate G Matrix
    for i in range(rows):
        G.append([])
        for j in range(cols):
            G[i].append(np.random.normal(mean, sd, None))
    
    # Calculate X1
    X1 = np.matmul(X, G)
    
    return(X1)
# End of rand_proj