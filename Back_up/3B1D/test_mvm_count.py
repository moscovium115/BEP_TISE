import numpy as np  # Example library (you would replace this with the library you're using)

# Create a custom matrix class that overrides the @ operator
class CustomMatrix(np.ndarray):  # Replace np.ndarray with the actual matrix class used by your library
    def __matmul__(self, other):
        # Increment the MVM counter
        global mvm_count
        mvm_count += 1

        # Perform the matrix-vector multiplication
        return super(CustomMatrix, self).__matmul__(other)

# Initialize a counter for MVMs
mvm_count = 0

# Your code that involves matrix operations
A = CustomMatrix([[1, 2], [3, 4]])
x = np.array([1, 2])

result = A @ x

# Print the total number of MVMs
print("Total MVMs:", mvm_count)
