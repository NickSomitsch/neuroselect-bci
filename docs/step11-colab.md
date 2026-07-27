# Step 11 Google Colab runbook

Step 11 evaluates all 3,990 held-out synthetic next-span trials with the pinned Qwen3-4B MLX
backend and four completed research adapters. Adapter training is already complete; Colab runs
inference and evaluation only.

## One-time local handoff

Commit the implementation first. The notebook deliberately refuses a moving branch or dirty
working tree. Then create the exact private-repository source bundle:

```bash
git add .
git commit -m "optimize Step 11 T4 inference"
git push
make language-cloud-source-bundle \
  LANGUAGE_CLOUD_SOURCE_BUNDLE_ARGS="--overwrite"
git rev-parse HEAD
```

The final command prints the 40-character SHA to paste into the notebook. The source archive is
`artifacts/cloud/neuroselect-step11-source.bundle`.

Create and verify the portable adapter/corpus inputs if the existing verified archive is not
already available:

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
MyDrive/neuroselect-step11/neuroselect-step11-source.bundle
```

Neither archive belongs in Git.

## Colab execution

Upload `notebooks/neuroselect_step11_colab.ipynb` in Colab, choose `Runtime > Change runtime type >
T4 GPU`, and run the cells in order.

The first cell rejects a missing GPU or CUDA compute capability below 7.5. A T4, L4, A100, or
newer compatible NVIDIA GPU is suitable. A P100 is not suitable for the MLX CUDA runtime.

Set only this required value in the configuration cell:

```python
GIT_REVISION = "<the exact pushed 40-character commit SHA>"
```

The notebook then:

1. mounts Google Drive;
2. clones the uploaded source bundle and detaches at that exact clean revision;
3. installs Python 3.12 plus the locked `local-language-cuda` optional dependencies;
4. runs a real MLX GPU calculation as a preflight;
5. verifies and safely extracts the 113 MB input archive;
6. downloads the configured `Qwen/Qwen3-4B-MLX-4bit` revision into
   `MyDrive/neuroselect-step11/huggingface-cache`, then copies it to the Colab local SSD;
7. runs a short four-profile optimized pilot and prints a full-run projection;
8. runs the full research protocol on local SSD with atomic Drive checkpoint mirrors; and
9. verifies and exports the canonical final result.

The pilot uses the development one-message-per-profile limit and the real research adapters. It
tests compatibility and provides an early speed estimate, but it is not research evidence. Do not
use the old pilot timing: the optimized runner batches continuation scoring with one shared prompt
cache, reuses identical context-only inference, enumerates the applicable train/validation-only
allow-list instead of asking Qwen to echo it as JSON, and reports a new projection. Qwen still
provides the measured likelihood ranking for every visible language candidate.

The T4 execution path preserves the research inputs and does not inspect or insert intended target
spans during candidate generation. For repeated identical confirmed contexts it reuses the same
generic or profile-conditioned evidence. The locked full workload contains 3,990 trials but only
1,908 unique generic contexts and about 1,981 unique profile/context combinations.

## Disconnects and resume

The full-run cell uses fast ephemeral storage for active writes:

```text
/content/neuroselect-step11-checkpoint/
```

Every 25 new trials are fsynced and atomically copied to:

```text
MyDrive/neuroselect-step11/checkpoint-optimized-v1/
```

If Colab disconnects, reconnect to a compatible GPU and rerun every notebook cell in order. The
full-run cell restores the Drive mirror to local SSD and skips completed trials. It refuses to
resume if any of these changed:

- Git revision or dirty source digest;
- research protocol or model configuration;
- generated benchmark or non-test candidate vocabulary;
- adapter or corpus manifests; or
- expected trial count.

At worst, an abrupt disconnect loses the fewer than 25 most recent unflushed trials. Never edit
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
