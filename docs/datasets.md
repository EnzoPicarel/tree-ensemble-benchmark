# Dataset sources and publication review

Reviewed 2026-10-23. No dataset files are distributed in this repository.

- **Fashion-MNIST:** Han Xiao, Kashif Rasul and Roland Vollgraf, *Fashion-MNIST:
  a Novel Image Dataset for Benchmarking Machine Learning Algorithms* (2017).
  [Official repository](https://github.com/zalandoresearch/fashion-mnist),
  [official MIT notice](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE),
  [paper](https://arxiv.org/abs/1708.07747). The official repository's license is MIT,
  copyright Zalando SE. The loader downloads public IDX files only on explicit request.
  Retain the upstream notice if redistributing upstream material. Its license does
  not grant rights to university starter code or this project's unrelated contents.
- **California Housing:** Pace, R. Kelley and Ronald Barry, *Sparse Spatial
  Autoregressions*, Statistics & Probability Letters 33 (1997), 291–297. Derived
  from the 1990 US census. Loaded through sklearn's fetch_california_housing;
  [official dataset description](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset),
  [StatLib source archive](https://lib.stat.cmu.edu/datasets/houses.zip).
  The reviewed description establishes provenance but does not establish an explicit
  redistribution license for this prepared dataset. Do not infer dataset rights
  from scikit-learn's software license. No redistribution is proposed here; permission
  for the intended public benchmark use still requires review before publication.
- **Quick/CART modes:** deterministic synthetic data generated using scikit-learn.
  They are clearly separated from historical full-data performance claims.

The source notebook used public benchmark data; no course download, handout, grading
instruction, private dataset or original prediction row is copied. Aggregate tables
are retained with file hashes. No legal clearance is claimed by this document.
