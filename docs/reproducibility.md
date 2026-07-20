# Reproducibility guide

## Environment and source

Use Python 3.12, `uv`, Node.js 22, and `pnpm` 10. Record the exact Git revision and keep the
worktree clean for release evidence. Install only from the tracked lockfiles:

```bash
make setup
make verify
make release-check
make package
```

CI performs the same formatting, lint, type, Python/UI test, release-metadata, package-build, and
frontend-build gates without downloading EEG or model artifacts.

## Experiment identity

Every experiment has a tracked YAML recipe, fixed seeds, immutable data/model/policy identifiers,
and a `RunManifest` containing runtime versions, device, output checksums, and dirty-tree state.
Never compare files before verifying their manifests. Hardware and library differences may still
change PyTorch optimization; do not silently pool runs from different environments.

Run the public controlled matrix with `make simulated-evaluation`. Study P and decoder commands are
documented in the README and require explicit local data preparation. Data-author `Train`/`Test`
folders are not model splits. Unknown labels never enter supervised metrics.

## Research report

After producing local artifacts, run:

```bash
make research-report
```

The tracked report recipe requires the controlled simulation and treats xDAWN, EEGNet, and
counterfactual artifacts as optional until locally available. It verifies every output checksum,
does not execute checkpoint files, reconstructs simulation result digests, and emits canonical
JSON, readable Markdown, and a report manifest. `release_ready` is false if a required source is
missing or if the tracked clean-source policy is violated.

Simulation intervals resample synthetic profiles and then paired trials. Counterfactual intervals
resample subjects and then trials. Evidence tiers are rendered in separate tables and must remain
separate in publications.

## Evidence hierarchy

1. Controlled simulation: engineering behavior only.
2. Original-task EEG: source P300 target/non-target decoding only.
3. Counterfactual replay: candidate-system behavior over remapped recorded evidence.
4. Future live or participant study: unavailable and requiring a separate protocol.

Absence of a higher tier cannot be filled by relabeling a lower tier.
