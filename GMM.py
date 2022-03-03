import numpy as np
from sklearn import mixture
from itertools import combinations

def cluster_weights(links, threshold):
    weights = np.transpose(np.array([links]))

    # Clustering links
    n = len(links)
    MIN_NUM_SAMPLES = 2
    if n > MIN_NUM_SAMPLES:
        # Fit a mixture of Gaussians with EM
        lowest_bic = np.infty
        bic = []
        global best_gmm
        n_components_range = range(2, n)
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                gmm = mixture.GMM(n_components=n_components, covariance_type='full')
                gmm.fit(weights)
                # Bayesian information criterion
                bic.append(gmm.bic(weights))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        # Averaging

        ids = best_gmm.predict(weights)
        print( best_gmm.n_components )
        total_weights = []
        unique_ids = list(set(ids))

        for i in unique_ids:
            c_ids = ids == i
            average = np.sum(links[c_ids]) / len(links[c_ids])
            links[c_ids] = average
            total_weights.append(np.sum(links[c_ids]))
        print('averaged : ', links)
        weak_cluster_ids = [i for w, i in zip(total_weights, unique_ids) if w < threshold]
        strong_clusters = [links[ids == i] for i in unique_ids if i not in weak_cluster_ids]
        """
        combos = []
        for cluster in strong_clusters:
            i = 1
            for e in cluster:
                if e * i < threshold:
                    combos.append(e)
                i += 1
        print('combos', combos)
        possible_combo = 0
        for i in range(0, len(combos) + 1):
            combo = set(combinations(combos, i))
            for c in combo:
                w = np.sum(list(c))
                if possible_combo < w < threshold:
                    possible_combo = w

        weak_clusters = [(total_weights[i], i) for i in weak_cluster_ids]
        for w, j in weak_clusters:
            weighted_combo = w + possible_combo
            if weighted_combo < threshold:
                idx = ids == j
                links[idx] = 0
        """
        print("output", links)
        print("clusters", ids)
        #print("weak", weak_clusters)
        print("strong", strong_clusters)
        return links, ids
    elif n == 2:
        return links, np.array([0, 1])
    else:
        return links, np.zeros(len(links))

#input = np.array([ 1.75419056,  0.54799098,  3.81015849])
#cluster_weights(input, 4.78619)
input = np.array([-1.47849345, -0.61190701, -0.00798364, 0.61219245, 1.47244668, 2.40201473])
cluster_weights(input, 999)
"""
[[-1.17434561 -0.80936354 -1.89170492 -1.78417099]
 [ 1.48889494 -0.80936354 -1.17912626 -0.03890961]
 [ 5.83259821 -0.80936354 -3.39527774  0.69538951]
 [-2.98253059 -0.80936354  3.81548238 -4.02680445]
 [-2.98253059 -0.80936354 -0.6557225  -1.78417099]
 [-4.88838148 -0.80936354  3.81548238 -0.03890961]]
"""

