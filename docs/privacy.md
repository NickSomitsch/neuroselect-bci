# Privacy statement

## Scope

NeuroSelect is a local-first research prototype. Setup and continuous integration do not download
EEG data or language models, and the application has no telemetry or cloud-upload integration.
The service is configured to bind only to loopback. This statement describes repository defaults,
not downstream deployments or third-party dataset hosts.

## Data handled

- Public Study P EEG and its subject/session/event metadata, when a researcher explicitly accepts
  the upstream license and downloads named files.
- Confirmed and provisional message text held in process memory for the local demo session.
- Local personal-knowledge records in SQLite, including their source, permissions, validity,
  profile, and revision history.
- Model checkpoints, decoder predictions, experiment tables, configuration, and environment
  provenance written under ignored local artifact directories.
- Four tracked public profiles and benchmark messages that are entirely synthetic.

The current demo does not persist sessions or message text after process exit. Experiment commands
persist only the artifacts they explicitly document. Uvicorn may emit ordinary request metadata
to the local terminal; do not enable verbose request-body logging when working with personal data.

## Control and deletion

Personal knowledge can be added, disabled, replaced, or physically deleted through the storage
layer. Raw datasets, derived EEG, reports, and checkpoints are ordinary local files and must be
deleted by the operator together with backups. A language-model LoRA importer/trainer is not yet
implemented; no adapter-deletion claim is made. Future adapter support must provide replacement
and deletion before it can accept personal text.

## Sharing and retention

Nothing is shared automatically. Before sharing an artifact, inspect it for message text, record
content, subject identifiers, filesystem paths, environment details, and licensing restrictions.
Run manifests intentionally retain provenance and can therefore reveal local platform and dataset
identities even when raw data is absent. Define a retention period before collecting personal or
human-subject data.

## Recommended safeguards

Use full-disk encryption, an encrypted backup with an independently tested deletion process, a
dedicated OS account for sensitive studies, restrictive filesystem permissions, and an offline or
firewalled environment where appropriate. Do not expose the API beyond loopback. Human-subject or
live-BCI studies require a separate consent process, ethics review, data-management plan, and
incident-response procedure.
