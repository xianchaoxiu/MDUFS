import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment



# ==========================================================
# Clustering accuracy
# ==========================================================

def clustering_acc(
        y_true,
        y_pred):

    """
    Clustering accuracy with Hungarian matching
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)


    assert (
        y_true.shape[0]
        ==
        y_pred.shape[0]
    )


    true_labels = np.unique(
        y_true
    )

    pred_labels = np.unique(
        y_pred
    )


    true_map = {
        lab:i
        for i,lab in enumerate(true_labels)
    }


    pred_map = {
        lab:i
        for i,lab in enumerate(pred_labels)
    }


    yt = np.vectorize(
        true_map.get
    )(y_true)


    yp = np.vectorize(
        pred_map.get
    )(y_pred)



    D = max(
        len(true_labels),
        len(pred_labels)
    )


    w = np.zeros(
        (D,D),
        dtype=np.int64
    )


    for i in range(
        yt.size
    ):

        w[
            yp[i],
            yt[i]
        ] += 1



    row_ind, col_ind = linear_sum_assignment(
        w.max()-w
    )


    return (
        float(
            w[row_ind,col_ind].sum()
        )
        /
        yt.size
    )



# ==========================================================
# KMeans ACC/NMI evaluation
# ==========================================================

def eval_kmeans_acc_nmi(
        X,
        y,
        n_clusters,
        n_runs=50,
        random_state_start=0):


    """
    Multiple KMeans runs.
    Return mean/std ACC and NMI.
    """


    accs = []
    nmis = []


    for r in range(n_runs):

        rs = (
            random_state_start
            +
            r
        )


        km = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            max_iter=300,
            random_state=rs
        )


        pred = km.fit_predict(
            X
        )


        accs.append(
            clustering_acc(
                y,
                pred
            )
        )


        nmis.append(
            normalized_mutual_info_score(
                y,
                pred
            )
        )



    accs = np.asarray(
        accs,
        dtype=np.float64
    )

    nmis = np.asarray(
        nmis,
        dtype=np.float64
    )



    return (

        accs.mean()*100,

        accs.std()*100,

        nmis.mean()*100,

        nmis.std()*100

    )