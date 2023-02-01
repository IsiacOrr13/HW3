import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)


def main():

    # create tight clusters
    clusters, labels = make_clusters(scale=0.3)
    plot_clusters(clusters, labels, filename="figures/tight_clusters.png")
    c1 = KMeans(3, 1e-6, 100)
    labels = c1.fit(clusters)
    plot_clusters(clusters, labels, filename="figures/tight_clusters_output.png")

    # create loose clusters
    clusters, labels = make_clusters(scale=2)
    plot_clusters(clusters, labels, filename="figures/loose_clusters.png")
    c2 = KMeans(3, 1e-6, 100)
    labels = c2.fit(clusters)
    plot_clusters(clusters, labels, filename="figures/loose_clusters_output.png")

    """
    uncomment this section once you are ready to visualize your kmeans + silhouette implementation
    """
    # clusters, labels = make_clusters(k=4, scale=1)
    # km = KMeans(k=4)
    # km.fit(clusters)
    # pred = km.predict(clusters)
    # scores = Silhouette().score(clusters, pred)
    # plot_multipanel(clusters, labels, pred, scores)
    

if __name__ == "__main__":
    main()
