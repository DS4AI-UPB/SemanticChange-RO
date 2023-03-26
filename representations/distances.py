from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import jensenshannon
import numpy as np

def compute_distance_for_points(a, b, requested_distance_metric):
    
    if "euclidean" in requested_distance_metric:
        return distance.euclidean(a, b) * 1e6

    if "cosine" in requested_distance_metric:
        return distance.cosine(a, b) * 1e6

    if "canberra" in requested_distance_metric:
        return distance.canberra(a, b) * 1e6
    
    if "braycurtis" in requested_distance_metric:
        return distance.braycurtis(a, b) * 1e6

    if "manhattan" in requested_distance_metric:
        return distance.cityblock(a, b) * 1e6

    if "correlation" in requested_distance_metric:
        return distance.correlation(a, b) * 1e6    

    return 0
    
def compute_distances_for_sets(a, b, requested_distance_metric):

    a = a.to_numpy()
    b = b.to_numpy()

    if requested_distance_metric == 'cluster_count':
        a_clustering = AffinityPropagation(random_state=9).fit(a)
        b_clustering = AffinityPropagation(random_state=9).fit(b)
        
        if requested_distance_metric == 'cluster_count':
            rez = 1 if np.max(a_clustering.labels_) != np.max(b_clustering.labels_) else 0
            return rez
        
    if requested_distance_metric.startswith('pointwise'):
        rez = [0]
        if requested_distance_metric == 'pointwise_euclidean':
            rez=pairwise_distances(a,b, metric='euclidean')
        if  requested_distance_metric == 'pointwise_cosine':
            rez=pairwise_distances(a,b, metric='cosine')
        if  requested_distance_metric == 'pointwise_canberra':
            rez=pairwise_distances(a,b, metric='canberra')
        if  requested_distance_metric == 'pointwise_jaccard':
            rez=pairwise_distances(a,b, metric='jaccard')
        if  requested_distance_metric == 'pointwise_manhattan':
            rez=pairwise_distances(a,b, metric='manhattan')
        
        max = np.max(rez)
        min = np.min(rez)
        mean = sum(sum(rez))/(len(rez)*len(rez[0]))
        if max != min:
            return (mean - min) / (max - min)
        else:
            return mean
    if requested_distance_metric == 'jsd':
        lena = len(a)
        lenb = len(b)
        if lena > lenb:
            a = a[:lenb]
        elif lenb > lena:
            b = b[:lena]
        rez = jensenshannon(a,b, axis=0)
        rez = list(filter(lambda x: x <= 1, rez))
        return sum(rez)/len(rez)
    return 0