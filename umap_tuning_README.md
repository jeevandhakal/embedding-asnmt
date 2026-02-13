**Dimension Reduction Analysis**

**Methodology**

1.  The UMAP algorithm's performance was evaluated by maximizing the
    average Spearman rank correlation between the high-dimensional
    (384D) cosine similarities and the 2D Euclidean distances. This
    metric ensures that the \"neighborhood\" of each student their most
    similar classmates is faithfully preserved in the visualization.

    **Assessments**

When the untuned model was run with different seeds it showed high
sensitivity. With not optimized hyperparameters the algorithm struggles
to resolve the complex 384D manifold, causing clusters to shift
significantly or even dissolve between runs. With optimized parameters
n_neighbors and min_dist the tuned model proved to be much stable. My
results across five different seeds showed a mean ranging from 0.58 to
0.70. This indicates that the optimization \"locked\" the model into a
configuration that prioritizes the underlying semantic structure over
random initialization noise.

**Performance**

The tuned model achieves a high degree of success in capturing student
interest patterns. An average of 0.62 is a strong result for social
data; it confirms that if two students appear close together in the png
files, there is a high mathematical probability that their interests are
truly aligned in the embedding space.

The fact that the model preferred a specific n_neighbors count suggests
the data contains distinct \"interest pockets\" (local structure) rather
than one continuous gradient. Overall, UMAP proved to be an adequate and
reliable tool for this identification task, successfully translating
abstract semantic similarities into a readable, trustworthy 2D map.
