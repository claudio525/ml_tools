from scipy.cluster import hierarchy


def compute_hierarchical_linkage_matrix(dist_mat, method='complete'):
    if method == 'complete':
        Z = hierarchy.complete(dist_mat)
    if method == 'single':
        Z = hierarchy.single(dist_mat)
    if method == 'average':
        Z = hierarchy.average(dist_mat)
    if method == 'ward':
        Z = hierarchy.ward(dist_mat)

    return Z