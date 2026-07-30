# Public repository boundary

This repository is designed to be safely cloneable without downloading EEG data, language-model
weights, or trained checkpoints. It contains research software and manuscript sources, not a
redistributable bundle of every local research artifact.

## Included

- Python and React/TypeScript source code, tests, locked dependency files, and CI;
- tracked experiment, model, dataset-inventory, publication, and release configurations;
- explicitly synthetic profiles, benchmark specifications, and public-safe fixtures;
- dataset and model cards, architecture decisions, responsible-use guidance, and reproducibility
  instructions;
- the target-blind Google Colab notebook with all outputs and execution counts cleared; and
- journal-neutral Markdown/LaTeX manuscript sources, structured BibTeX, and publication figures.

## Excluded

- raw or processed Study P EEG and any other downloaded dataset;
- Qwen or other base-model weights and caches;
- LoRA adapters, decoder checkpoints, joblib/PyTorch payloads, and intermediate checkpoints;
- generated experiment/report/manuscript bundles under `artifacts/`;
- local SQLite stores, personal messages, retrieval records, terminal logs, screenshots, and
  environment overrides; and
- access tokens, private keys, credentials, and consumer email addresses.

These classes are ignored by Git and checked again by `make public-release-check`. Upstream
datasets, model weights, and derived artifacts retain their own licenses even when a script can
download or produce them.

## Before changing visibility

Run from a clean checkout:

```bash
make public-release-check
make verify
make release-check
git status --short
```

The public audit checks current public-candidate files and the complete available Git history for
restricted paths, credential patterns, private home-directory paths, consumer email addresses,
executed notebook state, and files larger than 1 MiB, including historical blobs. GitHub Actions
uses a full-history checkout so the same audit remains meaningful in CI.

Git commit author and committer addresses are public metadata. Inspect them separately before
publication:

```bash
git log --all --format='%ae%n%ce' | sort -u
```

Changing an existing address requires a coordinated history rewrite and force-push; do not do that
casually after collaborators have cloned the project.

After the cleanup commit is pushed, confirm that the default branch passes CI and then change the
repository visibility. Enable private vulnerability reporting, secret scanning, and branch
protection or rulesets where the GitHub plan permits them.

Changing visibility can also expose existing Actions logs, issues, pull requests, releases, and
other GitHub-hosted metadata. Review those surfaces separately; the local audit cannot inspect
them.

## After publication

- Never attach excluded artifacts to an issue or pull request.
- Keep GitHub Actions permissions read-only unless a narrowly scoped workflow requires more.
- Review dependency and CodeQL alerts before releases.
- Re-run the public audit before tags, archived research snapshots, or manuscript supplements.
- Treat a public repository as irrevocably copied: removing a file later does not remove existing
  clones, forks, caches, or release archives.
