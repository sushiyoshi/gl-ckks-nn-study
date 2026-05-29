from desilofhe import GLEngine
import numpy as np
engine = GLEngine()

secret_key = engine.create_secret_key()
matrix_multiplication_key = engine.create_matrix_multiplication_key(secret_key)
ciphertext1 = engine.encrypt(np.ones(engine.shape), secret_key)
ciphertext2 = engine.encrypt(np.ones(engine.shape), secret_key)

multiplied = engine.matrix_multiply(
    ciphertext1, ciphertext2, matrix_multiplication_key
)
