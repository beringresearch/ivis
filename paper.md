---
title: 'ivis: dimensionality reduction in very large datasets using Siamese Networks'
tags:
  - dimensionality reduction
  - unsupervised learning
  - neural network
authors:
  - name: Benjamin Szubert
    affiliation: 1
  - name: Ignat Drozdov
    affiliation: 1
    orcid: 0000-0001-6727-4688
affiliations:
  - name: Bering Limited
    index: 1
date: 18 July 2019
bibliography: paper.bib
---

# Summary

`ivis` is a dimensionality reduction technique that implements a Siamese Neural Network architecture trained using a novel triplet loss function. Results on simulated and real datasets demonstrate that `ivis` preserves global data structures in a low-dimensional space and adds new data points to existing embeddings using a parametric mapping function. 

`ivis` is easily integrated into standard machine learning pipelines through a scikit-learn compatible API and scales well to out-of-memory datasets. Both supervised and unsupervised dimensionality reduction modes are supported.

Further information on the algorithm and its application to single cell datasets can be found in [@ivis_scirep]. Implementation of the `ivis` algorithm is available on [GitHub](https://github.com/beringresearch/ivis).

# Acknowledgements
This work was supported by funding from the European Commission’s Seventh Framework Programme [FP7-2007-2013] under grant agreement n°HEALTH-F2-2013-602114 (Athero-B-Cell).

# References
