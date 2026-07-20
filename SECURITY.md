# Security policy

NeuroSelect is pre-release research software. It is not hardened for clinical deployment,
multi-user hosting, or exposure to an untrusted network.

## Reporting a vulnerability

Use the repository's private GitHub Security Advisory reporting channel. Do not include raw EEG,
personal messages, access tokens, personal RAG records, or other sensitive data in a public issue.
If private reporting is unavailable, open a public issue containing only a request for a private
contact channel.

Include the affected revision, reproduction steps using synthetic data, impact, and any proposed
mitigation. There is currently no guaranteed response-time service level.

## Supported scope

Only the current default branch is maintained. The API must remain bound to loopback. Do not place
it behind a public reverse proxy or use it as a shared service. Treat joblib decoder checkpoints as
executable content and load only trusted, locally produced files. PyTorch checkpoints must retain
the tensor-only, weights-only loading path.

See [the threat model](docs/threat-model.md) for trust boundaries and known residual risks.
