"""
Tier 6: Domain Adaptation Utilities (CORAL & TCA)
=================================================
Provides standalone mathematical projections to align source and target domains.
"""

import numpy as np
import scipy.linalg

def apply_coral(source_x, target_x):
    """
    CORAL (Correlation Alignment) algorithm.
    Projects source_x to have the same covariance as target_x.
    """
    # Small regularization
    epsilon = 1.0
    
    # Dimensions
    ns = source_x.shape[0]
    nt = target_x.shape[0]
    
    if ns < 2 or nt < 2:
        return source_x

    # Calculate covariances
    cs = np.cov(source_x, rowvar=False) + np.eye(source_x.shape[1]) * epsilon
    ct = np.cov(target_x, rowvar=False) + np.eye(target_x.shape[1]) * epsilon

    try:
        # Whitening Source
        cs_inv_sqrt = scipy.linalg.fractional_matrix_power(cs, -0.5)
        # Coloring with Target
        ct_sqrt = scipy.linalg.fractional_matrix_power(ct, 0.5)
        
        # We need real matrices
        if np.iscomplexobj(cs_inv_sqrt): cs_inv_sqrt = cs_inv_sqrt.real
        if np.iscomplexobj(ct_sqrt): ct_sqrt = ct_sqrt.real
            
        trans_matrix = np.dot(cs_inv_sqrt, ct_sqrt)
        
        # Center, Transform, Re-center to target
        source_centered = source_x - np.mean(source_x, axis=0)
        source_aligned = np.dot(source_centered, trans_matrix)
        source_aligned += np.mean(target_x, axis=0)
        
        return source_aligned
    except Exception as e:
        print(f"CORAL inversion failed: {e}. Returning original.")
        return source_x


def apply_tca(source_x, target_x, target_val, target_test, n_components=10, max_samples=5000):
    """
    TCA (Transfer Component Analysis) - Linear Approximation for scalability.
    Since O(N^3) eigenvalue decomposition on millions of windows will OOM, 
    we sub-sample to find mapping, then project everything.
    For simplicity and speed in GBM pipeline, we implement a linear PCA-based MMD.
    """
    # Subsample to avoid memory crash
    idx_s = np.random.choice(len(source_x), min(len(source_x), max_samples), replace=False)
    idx_t = np.random.choice(len(target_x), min(len(target_x), max_samples), replace=False)
    
    Xs = source_x[idx_s]
    Xt = target_x[idx_t]
    
    X = np.vstack([Xs, Xt])
    
    # MMD Matrix L
    ns, nt = len(Xs), len(Xt)
    e = np.vstack([1.0/ns * np.ones((ns, 1)), -1.0/nt * np.ones((nt, 1))]).reshape(-1, 1)
    L = np.dot(e, e.T)
    
    # Linear Kernel (X X^T)
    K = np.dot(X, X.T)
    
    # Centering matrix H
    H = np.eye(ns + nt) - 1.0 / (ns + nt) * np.ones((ns + nt, ns + nt))
    
    # Optimization: I + mu * K * L * K
    mu = 1.0
    Kc = np.dot(np.dot(H, K), H) # Centered kernel
    A = np.eye(ns + nt) + mu * np.dot(np.dot(Kc, L), Kc)
    B = Kc
    
    try:
        # Solve generalized eigenvalue problem: B * w = lambda * A * w
        # We invert A for simplicity since A is pos-def
        A_inv_B = np.dot(np.linalg.inv(A), B)
        eigenvalues, eigenvectors = scipy.linalg.eigh(A_inv_B)
        
        # Take top n_components (largest eigenvalues are at the end)
        W = eigenvectors[:, -n_components:]
        
        # Projection of original full data
        # Linear approximation: For test data x, projection is (x * X^T) * W
        mapping = np.dot(X.T, W)  
        source_mapped = np.dot(source_x, mapping)
        target_val_mapped = np.dot(target_val, mapping)
        target_test_mapped = np.dot(target_test, mapping)
        
        return source_mapped, target_val_mapped, target_test_mapped
    except Exception as e:
        print(f"TCA decomposition failed: {e}. Returning PCA reduction.")
        # Fallback to PCA if singular
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(source_x)
        return pca.transform(source_x), pca.transform(target_val), pca.transform(target_test)
