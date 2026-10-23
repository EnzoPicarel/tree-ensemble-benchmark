# Provenance and scope

This is a separate portfolio edition of a University of Cincinnati machine-learning
coursework study by Enzo Picarel. Its initial commit is new; no academic Git history
is imported. Source history is summarized in [source-history.txt](source-history.txt).
The source notebook SHA-256 and original commit are recorded in the
[artifact manifest](../results/historical/artifact_manifest.json).

The NumPy CART algorithm was extracted from source notebook cell 71. Its greedy
sorted-feature searches, impurity calculations, zero-gain split behavior, stopping
rules, class encoding and prediction logic are preserved. Portfolio changes reject
boolean integer parameters and nonfinite numeric labels, and make unfitted predict
raise the intended ValueError before accessing classes_. Cells 72's fixtures and
90-configuration synthetic comparison inform the standalone tests and CART command.

The OOF procedure follows cells 58 and 63–67: five folds on training rows, logistic
classification / ridge regression meta-models, and final base fits on the original
training partition without merging validation. Frozen model settings come from the
source's pre-test manifest, converted to explicit JSON without evaluating source code.
Every tree ensemble in the full-data benchmark is a scikit-learn estimator; custom
CART is evaluated separately on small synthetic datasets.

New portfolio work includes modules, CLI, strict OOF partition validation, tests,
metadata, output protection, package configuration, and regenerated presentation.
Historical CSVs remain byte-for-byte copies. A separately recorded Housing seed-42
replay reproduces that seed's error/node metrics, and the small 90-configuration
CART study was rerun; the five-seed main campaign is still historical. New timings use a newly documented
measurement path; they must not be presented as the original timing campaign.

Commit attribution supports Enzo Picarel's participation; it does not establish
exclusive authorship or prove that no instructor-provided starter code was used.
No collaborator or starter-code license was found establishing unrestricted
publication. Retain this uncertainty pending owner/instructor confirmation.
AI assistance was used in reorganizing and testing this portfolio edition.
No assignment handouts, grading instructions, source notebook, student datasets,
per-example predictions, pretrained models, or academic Git directory are included.
