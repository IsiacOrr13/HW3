import pytest
from cluster import (
        KMeans,
        Silhouette,
        make_clusters,
        plot_clusters,
        plot_multipanel)

clusters, labels = make_clusters(k=4, scale=1)

def test_label_size():
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    assert len(pred) == 500

def test_init():
    km = KMeans(k=4)
    assert isinstance(km, object)

def test_return():
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    assert isinstance(pred, object)