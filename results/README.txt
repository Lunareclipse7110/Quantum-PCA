This directory is populated automatically when you run the project:

    python main.py            (or python main.py --quick for a fast test)

It will contain:
  results.csv                       - every individual experiment run
  results_aggregated.csv            - mean +/- std per (dataset, experiment)
  training_curves/*.json            - COBYLA cost-vs-iteration for one repeat per config
  example_counts/*.json             - one sample's measurement counts per config
  figures/*.png                     - all required plots (see main README.md, section 6)
  Team4_Quantum_PCA_Report.html     - THE ONE TO OPEN: a single presentable
                                       report page with everything above laid
                                       out, tables + figures + write-up. Open
                                       it in any browser; print to PDF for a
                                       hand-in copy.
