
# coding: utf-8

# ****input a matrix
# **** and output a matrix
# SVD is verified

# In[8]:


#dimension reduction




import numpy as np
from sklearn.decomposition import TruncatedSVD
def svd_dim_reduction(matrix):
    svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42)
    matrix_transform = svd.fit_transform(matrix)
    return matrix_transform
#test code
# a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
# print(svd_dim_reduction(a))


