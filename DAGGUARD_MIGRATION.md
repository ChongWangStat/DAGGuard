# Project naming migration

The method and public-facing software are named **DAGGuard**. The GitHub repository was renamed from `NOTEARS-BP` to `DAGGuard` on August 24, 2026.

The application analyses used in the manuscript remain tied to historical commit `509cb29c24967d12b99e2b53641349ce7bb470ed`, created before the repository rename. Renaming the repository does not change that commit hash, and GitHub redirects the earlier repository URL to the current DAGGuard repository. Keeping the historical commit identifiable preserves the reproducibility audit trail without requiring old analysis files to be rewritten.

New analyses should use:

```python
from dagguard import refine_dag, pruning_pressure
```

The legacy module `local_bic_refinement.py` remains the tested numerical engine and is retained for backward compatibility with the historical reproducibility workflow. The current public repository is:

`https://github.com/ChongWangStat/DAGGuard`
