# Interview preparation

- **What did you implement?** NumPy CART and the experimental/OOF orchestration.
  Full-data ensemble algorithms are provided by scikit-learn. Be explicit about
  coursework provenance and AI-assisted portfolio refactoring; confirm starter-code
  ownership before claiming exclusive authorship.
- **How does CART search?** Sort each numeric feature, accumulate class counts,
  evaluate weighted child impurities at distinct adjacent values, and choose the
  largest decrease. Depth/leaf limits stop growth. Equal-gain choices can differ
  from sklearn. A zero-gain XOR root must remain eligible for a useful second split.
- **Why OOF?** A meta-model trained on in-sample base predictions would see an
  unrealistically easy learning problem. Each OOF row is predicted by a member
  trained on other folds. The classification stack has 40 probability features;
  regression has four scalar features. Members are then fit on all training rows.
- **What does five-seed SD mean?** Estimator randomness conditional on a fixed split.
  It is not uncertainty about new locations, new time periods, or independent draws.
- **Why not always stacking?** Historical housing error improves only about 0.29%,
  at 13.33x recorded fitting and 15.59x recorded prediction cost. Deployment constraints
  could favor gradient boosting; these timings are machine- and protocol-specific.
- **Are the evaluations unbiased?** Broad validation reuse can bias selection, and
  the pruning candidate path was not derived inside each CV fold. Frozen replay
  preserves the original selection without claiming to fix those old estimates.
  Housing's geographic mixture is not unseen-region validation.
- **What does model size mean here?** Node counts plus separate meta-parameter counts;
  not file bytes or peak memory. OOF trees add training cost but are discarded before
  inference. Summed component latency is not a warmed serving benchmark.
- **What was verified?** Read docs/verification.md. Distinguish newly executed tests
  and synthetic/partial-data reruns from historical five-seed results. An offline
  synthetic quick run proves execution and safeguards, not real-data performance.
