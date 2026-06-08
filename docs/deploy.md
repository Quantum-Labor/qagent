# Deploying the Hugging Face Space

The live demo at
[huggingface.co/spaces/Laborator/qagent](https://huggingface.co/spaces/Laborator/qagent)
is a mirror of the `space/` directory in this repository. The mirror is produced
by the GitHub Actions workflow
[`.github/workflows/deploy-space.yml`](../.github/workflows/deploy-space.yml),
which uploads `space/` from `main` to the Space repo via `HfApi.upload_folder`.

Source of truth lives here; the Space repo is a deploy target and should not be
edited directly.

## What gets deployed

| Source in this repo | Destination at the Space root |
| --- | --- |
| `space/` (app.py, safety.py, README.md, requirements.txt, Dockerfile, assets/, precomputed/) | repo root |

Everything the Space needs lives under `space/`, including `precomputed/`
(the baked QAOA/greedy/brute-force results and the source dataset for live
verification) and `assets/` (logo, hero SVG, CSS). The Space's `.gitattributes`
(Git-LFS config) is **not** overwritten by the sync; it exists only on the HF
side.

## Why the HTTP API, not git push

git-over-HTTPS push to the Space is rejected from GitHub Actions runners
regardless of the token or credential mechanism. The workflow therefore uploads
the assembled folder through `huggingface_hub.HfApi.upload_folder` (Bearer auth),
which is the path that worked reliably for the QVerify Space.

## One-time setup: the `HF_TOKEN` secret

The workflow needs a Hugging Face token with write access to the Space.

1. Mint the token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
   **New token**, scope **Write** (or a fine-grained token granted write access to
   the `Laborator/qagent` Space). Copy the `hf_...` value (shown only once).
2. Add it as a GitHub repository secret at
   `github.com/Quantum-Labor/qagent/settings/secrets/actions`: **New repository
   secret**, name `HF_TOKEN`, value the token. The same token used for the QVerify
   Space works for any Space owned by `Laborator`.

## When it runs

- **Automatically** on every push to `main` that touches `space/**` (the `paths`
  filter). Pushes that change only `qagent/`, `tests/`, or docs do not deploy.
- **Manually**: GitHub repo -> **Actions** -> **Deploy to HF Space** ->
  **Run workflow** -> pick `main`.

## Updating the precomputed results

The served QAOA numbers are baked by `scripts/precompute_results.py`
(full QAOA per task, ~1h on CPU). Re-run it, commit the regenerated
`space/precomputed/benchmark_results.json`, and the next push to `main` deploys
it.

## Rollback

The deploy is a pure mirror, so rolling back the Space means reverting the
offending commit on `main` and letting the workflow re-sync, or re-running the
workflow from an earlier commit via **Run workflow**.
