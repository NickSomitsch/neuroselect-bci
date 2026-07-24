"""Held-out-safe candidate vocabulary derived without reading test messages."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuroselect.synthetic import BenchmarkSplit, GeneratedBenchmark

CANDIDATE_VOCABULARY_REVISION: Literal["non-test-autocomplete-v1"] = "non-test-autocomplete-v1"


def _normalized(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = " ".join(text.split())
    return text.rstrip(".,!?;:")


class HeldOutCandidateVocabulary(BaseModel):
    """Compact phrase classes learned only from train and validation messages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    vocabulary_revision: Literal["non-test-autocomplete-v1"] = CANDIDATE_VOCABULARY_REVISION
    benchmark_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_splits: tuple[Literal["train"], Literal["validation"]] = (
        "train",
        "validation",
    )
    noun_phrases: tuple[str, ...] = Field(min_length=1)
    deadline_phrases: tuple[str, ...] = Field(min_length=1)
    ending_phrases: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_bounded_phrases(self) -> HeldOutCandidateVocabulary:
        for label, phrases in (
            ("noun", self.noun_phrases),
            ("deadline", self.deadline_phrases),
            ("ending", self.ending_phrases),
        ):
            keys = [_normalized(phrase) for phrase in phrases]
            if any(
                not key or len(phrase.split()) > 4
                for key, phrase in zip(keys, phrases, strict=True)
            ) or len(keys) != len(set(keys)):
                raise ValueError(
                    f"{label} candidate vocabulary must contain unique 1-4-token phrases"
                )
        return self

    def phrases_for(self, confirmed_text: str) -> tuple[str, ...]:
        """Select a phrase class from visible context alone."""

        context = _normalized(confirmed_text)
        if not context:
            return ()
        if any(
            re.search(rf"(?<!\w){re.escape(_normalized(phrase))}(?!\w)", context)
            for phrase in self.deadline_phrases
        ):
            return self.ending_phrases
        if any(context.endswith(_normalized(phrase)) for phrase in self.noun_phrases):
            return self.deadline_phrases
        return self.noun_phrases

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def build_held_out_candidate_vocabulary(
    benchmark: GeneratedBenchmark,
) -> HeldOutCandidateVocabulary:
    """Build candidate phrase classes while structurally excluding the test split."""

    messages = (
        *benchmark.messages[BenchmarkSplit.TRAIN],
        *benchmark.messages[BenchmarkSplit.VALIDATION],
    )
    spans = {span for message in messages for span in message.target_spans}
    noun_phrases = {
        span
        for span in spans
        if re.match(r"^(?:the|my|today's)\s", span, re.IGNORECASE)
        and not span.endswith((".", "?", "!"))
        and len(span.split()) <= 3
    }
    deadline_phrases = {
        f"{left} {right}".strip(".,!?;:")
        for message in messages
        for left, right in zip(message.target_spans, message.target_spans[1:], strict=False)
        if left.casefold() in {"before", "by"}
        and not right.casefold().startswith(("before ", "after "))
        and len(f"{left} {right}".split()) <= 4
    }
    ending_phrases = {
        span for span in spans if span.endswith((".", "?", "!")) and len(span.split()) <= 4
    }
    return HeldOutCandidateVocabulary(
        benchmark_source_sha256=benchmark.source_sha256,
        noun_phrases=tuple(sorted(noun_phrases, key=str.casefold)),
        deadline_phrases=tuple(sorted(deadline_phrases, key=str.casefold)),
        ending_phrases=tuple(sorted(ending_phrases, key=str.casefold)),
    )
