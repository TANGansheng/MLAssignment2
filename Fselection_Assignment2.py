
# coding: utf-8

# ****input a matrix
# **** and output a matrix
# SVD is verified

# In[19]:


#dimension reduction

#SVD
import numpy as np
from sklearn.decomposition import TruncatedSVD
def svd_dim_reduction(matrix,n_components):
    svd = TruncatedSVD(n_components, n_iter=7, random_state=42)
    matrix_transform = svd.fit_transform(matrix)
    return matrix_transform
#test code
# a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
# print(svd_dim_reduction(a,3))

#PCA
from sklearn.decomposition import PCA
def pca_dim_reduction(matrix,n_components):
    pca = PCA(n_components)
    matrix_transform = pca.fit_transform(matrix)
    return matrix_transform
#test code
#a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
#print(pca_dim_reduction(a,2))

#feature selection
from sklearn.feature_selection import chi2
def chi2_featureselection(matrix, label):
    chi2val,pval=chi2(matrix,label)
    return [chi2val,pval]
#test code
# a = np.random.randint(low=1, high=10, size=(9,6), dtype='l')
# print(a)
# b=np.array([1,0,1,0,1,1,1,0,0])
# print(chi2_featureselection(a,b))

