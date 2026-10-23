# Publication gate: BLOCKED

The local engineering edition can be reviewed and tested. It must not be published,
licensed, or pushed until the following ownership questions are resolved:

1. Confirm university/course policy permits public release of the derived implementation
   and aggregate results. A private academic remote and commits do not establish permission.
2. Confirm whether CART/experimental components include starter code or collaborators'
   contributions, and obtain required attribution/permission. Source history alone
   cannot settle this. Enzo Picarel is credited for the study; exclusive authorship
   is not asserted.
3. Confirm California Housing's prepared-data terms support the intended use and
   attribution. Public accessibility and sklearn's software license are insufficient.

Fashion-MNIST's official repository MIT notice was reviewed; dataset files are not
redistributed. Source URLs and attribution are in datasets.md. No project LICENSE
is added because ownership is unresolved. No remote repository is created, no
visibility setting is changed, and no push is performed.

CI is withheld while publication is blocked. A minimal prospective workflow template
is in ci-template.txt; it runs pytest only. Activate it as .github/workflows/tests.yml
once the gate is cleared. Full dataset training must never run in CI.

Local checks cover tracked text for credential patterns, private paths, restricted
file types, large artifacts and Markdown links. They do not guarantee absence of
all possible confidential information. Results/historical contains aggregate tables,
not input data, model checkpoints or per-example predictions. Source paths and the
before/after source fingerprint are kept only in ignored .audit/ local records.
