# V0.1 Kaggle Smoke Test

The `python-smoke` profile is dependency-free and exists only to validate the end-to-end bridge:

`ChatGPT -> GitHub Issue -> GitHub Actions -> Kaggle -> status/evidence -> ChatGPT`.

It uses only the Python standard library, creates `smoke-result.json`, and does not require internet or GPU.

A successful smoke proves credential injection, private source packaging, Kaggle submission, execution, and result collection independently of project dependencies.
