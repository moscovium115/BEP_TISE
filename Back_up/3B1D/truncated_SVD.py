import numpy as np
from sklearn.decomposition import TruncatedSVD

# Create a sample matrix (replace this with your actual matrix)
#4 x4 matrix
matrix=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])

# Choose the number of components (k) to retain
k = 2

# Create a TruncatedSVD object and fit it to your matrix
svd = TruncatedSVD(n_components=k)
matrix_reduced = svd.fit_transform(matrix)

# Reconstruct the original dimensions by transforming back
matrix_restored = svd.inverse_transform(matrix_reduced)

print("Original Matrix:")
print(matrix)
print("\nReduced Matrix (Truncated SVD):")
print(matrix_reduced)
print("\nRestored Matrix (Original Dimensions):")
print(matrix_restored)
