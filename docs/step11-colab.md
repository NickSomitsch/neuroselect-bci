# Step 11 Google Colab runbook

Step 11 evaluates all 3,990 held-out synthetic next-span trials with the pinned Qwen3-4B MLX
backend and four completed research adapters. Adapter training is already complete; Colab runs
inference and evaluation only.

## One-time local handoff

Create and verify the portable inputs:

```bash
make language-cloud-bundle LANGUAGE_CLOUD_BUNDLE_ARGS="--overwrite"
make language-cloud-verify \
  LANGUAGE_CLOUD_BUNDLE=artifacts/cloud/step11-language-inputs-v1.tar.gz
```

The archive is written to `artifacts/cloud/step11-language-inputs-v1.tar.gz`. It is ignored by Git
and must not be committed. It contains:

- `adapters.safetensors`, `adapter_config.json`, and the provenance manifest for each of the four
  `-research-v1` adapters; and
- the train, validation, and test JSONL corpora plus manifest for each profile.

It excludes all adapter training checkpoints and the Qwen weights. The bundle creator checks the
adapter weights, corpora, model/revision, training tier, validation/test evaluation flags, and
adapter-to-corpus provenance before writing a deterministic archive.

Upload the archive to:

```text
MyDrive/neuroselect-step11/step11-language-inputs-v1.tar.gz
```

Commit and push the implementation, then copy the exact 40-character commit SHA. The notebook uses
a detached checkout because a moving branch cannot safely resume a research run.

## Colab execution

Open `notebooks/neuroselect_step11_colab.ipynb` in Colab and choose a GPU runtime. Run the cells in
order.

The first cell rejects a missing GPU or CUDA compute capability below 7.5. A T4, L4, A100, or
newer compatible NVIDIA GPU is suitable. A P100 is not suitable for the MLX CUDA runtime.

Set only this required value in the configuration cell:

```python
GIT_REVISION = "<the exact pushed 40-character commit SHA>"
```

The notebook then:

1. mounts Google Drive;
2. clones and detaches at that exact clean revision;
3. installs Python 3.12 plus the locked `local-language-cuda` optional dependencies;
4. runs a real MLX GPU calculation as a preflight;
5. verifies and safely extracts the 113 MB input archive;
6. downloads the configured `Qwen/Qwen3-4B-MLX-4bit` revision into
   `MyDrive/neuroselect-step11/huggingface-cache`;
7. runs a short four-profile pilot;
8. runs the full research protocol with durable checkpoints; and
9. verifies and exports the canonical final result.

The pilot uses the development one-message-per-profile limit and the real research adapters. It
tests compatibility and provides an early speed estimate, but it is not research evidence.

## Disconnects and resume

The full-run cell uses:

```text
MyDrive/neuroselect-step11/checkpoint-v1/
```

Every five new trials are flushed and synced. If Colab disconnects, reconnect to a compatible GPU
and rerun the notebook cells. The same full-run cell skips completed trials. It refuses to resume
if any of these changed:

- Git revision or dirty source digest;
- research protocol or model configuration;
- generated benchmark or non-test candidate vocabulary;
- adapter or corpus manifests; or
- expected trial count.

At worst, an abrupt disconnect loses the fewer than five most recent unflushed trials. Never edit
`checkpoint.json` or `trials.jsonl`. If intentionally starting with different inputs, choose a new
checkpoint directory instead of reusing the old one.

## Completion and import

A successful run produces:

```text
MyDrive/neuroselect-step11/held-out-language-personalization-research-v1/
MyDrive/neuroselect-step11/held-out-language-personalization-research-v1.tar.gz
```

The strict verifier requires:

- the locked research spec and pinned model metadata;
- exactly 3,990 unique, ordered teacher-forced spans;
- the four current research adapter and corpus checksums;
- the non-test-only vocabulary checksum;
- a clean producing Git revision;
- checksum-valid result, trial, metric, and run manifests; and
- claim eligibility derived from full coverage and research-trained adapters.

After copying the export back to the repository and checking out the producing commit, run:

```bash
make language-research-verify
```

Passing Step 11 supplies the complete held-out language component evidence. The next project step
is to rebuild the counterfactual research input from this result, run the claim-eligible
counterfactual fusion evaluation, and then regenerate the research evidence report.
