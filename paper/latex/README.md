# NeuroSelect LaTeX manuscript

`neuroselect-offline-journal-manuscript.tex` and `references.bib` are synchronized, tracked
journal-editable forms of the verified manuscript inputs. Do not hand-edit them while
`paper/manuscript.md` remains the canonical journal-neutral prose.

After changing the Markdown, reference registry, publication tables, or publication figures, run:

```bash
make manuscript-latex-sync MANUSCRIPT_ARGS="--overwrite --allow-dirty"
```

Inspect the resulting diff and compiled PDF, commit the synchronized files, and then create the
clean bundle:

```bash
make manuscript MANUSCRIPT_ARGS="--overwrite"
```

Tectonic is required:

```bash
brew install tectonic
```

The tracked package and the assembled artifact bundle each include the same checksum-verified
vector PDF figures under `figures/`; both forms are self-contained.
