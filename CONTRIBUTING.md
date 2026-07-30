# Contributing to NeuroSelect

NeuroSelect welcomes reproducibility, accessibility, safety, documentation, and research-method
improvements. Contributions must preserve the boundary between neural evidence, language-model
support, personalization, retrieval, and explicit user confirmation.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Install Python 3.12, `uv`, Node.js 22, and `pnpm` 10, then run:

```bash
make setup
make public-release-check
make verify
make release-check
```

Use focused tests while developing and run `make verify` before submitting a change. Install the
tracked hooks with `uv run pre-commit install` if desired. Pull requests should describe the
changed claim scope, data boundary, configuration or schema revision, and tests performed.

## Research changes

- Never describe simulated probabilities as EEG, counterfactual candidate text as participant
  intent, or an aggregate original-task metric as candidate communication performance.
- Keep train, calibration, test, subject, session, recording, and selection-trial boundaries
  explicit. New experiments require tracked configuration and checksum-addressed artifacts.
- Add uncertainty summaries where the sample supports them and state the resampling unit.
- Do not weaken abstention, enhanced confirmation, explicit final confirmation, or the
  `automatic_selection_permitted=false` invariant without an accepted architecture decision.
- Update the relevant model card, dataset card, limitations, and protocol ADR when scope changes.

## Data and privacy

Do not commit raw EEG, downloaded datasets, checkpoints, personal messages, personal knowledge
records, secrets, or generated artifacts containing private data. Public fixtures must be clearly
synthetic. Use `data/`, `models/`, and `artifacts/`, which are ignored by Git. Security issues or
accidental sensitive-data exposure should follow [SECURITY.md](SECURITY.md), not a public issue.

By contributing, you agree that your contribution is licensed under the repository's MIT license.
