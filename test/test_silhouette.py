# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster import (
        KMeans,
        Silhouette,
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_silhouette_return():
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    assert len(pred) == len(scores)
    assert isinstance(scores, object)

def test_silhouette():
    clusters, labels = make_clusters(k=4, scale=1)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    scores = scores.tolist()
    assert scores[0] == [0.8776099221466138]

