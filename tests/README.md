# tests — evaluation gates

These tests decide whether a number can be trusted. The anti-leakage tests arrive
with the data layer and run in CI on every pull request. A pull request that breaks
them does not merge.

```bash
python -m pytest tests/
```
