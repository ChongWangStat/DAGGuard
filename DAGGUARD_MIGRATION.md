# Project naming migration

The method and public-facing software are now named **DAGGuard**.

The historical repository slug and pinned reproducibility commit use the earlier NOTEARS-BP name. To preserve permanent reproducibility, those files and commit identifiers are not rewritten. New analyses should use:

```python
from dagguard import refine_dag, pruning_pressure
```

The legacy module `local_bic_refinement.py` remains the numerical engine and is retained for backward compatibility with the published reproducibility history. The repository can be renamed from `NOTEARS-BP` to `DAGGuard` in GitHub Settings without changing existing commit hashes; GitHub normally redirects the old URL after a rename.
