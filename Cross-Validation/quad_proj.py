import numpy as np

# Function to project input matrix X of dimensions mxn 
# to a matrix of dimension m x (2n + (n*(n-1))/2)

def quad_proj(X):
    # Init
    X2 = []
    rows = len(X)
    cols = len(X[0])
    
    # Calculate X2 Matrix
    for i in range(rows):
        X2.append([])
        for j in range(cols):
            X2[i].append(X[i][j])
        
        for j in range(cols):
            X2[i].append(X[i][j]**2)
        
        for k in range(cols):
            for j in range(k+1, cols):
                X2[i].append(X[i][k]*X[i][j])
    
    return(X2)
# End of rand_proj