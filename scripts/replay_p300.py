"""Emit a prepared Study P recording through the deterministic virtual replay clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroselect.bci import EpochReplay, ProbabilityDecoder, ReplayState
from neuroselect.decoding import read_decoder_artifacts, read_eegnet_artifacts
from neuroselect.eeg import read_epoch_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch_directory", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seek-seconds", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")
    decoder: ProbabilityDecoder | None = None
    if args.checkpoint:
        if (args.checkpoint / "eegnet.json").exists():
            decoder = read_eegnet_artifacts(args.checkpoint)[0]
        elif (args.checkpoint / "decoder.json").exists():
            decoder = read_decoder_artifacts(args.checkpoint)[0]
        else:
            raise SystemExit("checkpoint directory is neither an EEGNet nor classical run")
    replay = EpochReplay(read_epoch_batch(args.epoch_directory), speed=args.speed, decoder=decoder)
    if args.seek_seconds:
        replay.seek_seconds(args.seek_seconds)
    if replay.state is ReplayState.FINISHED:
        raise SystemExit("seek position is after the final replay event")
    replay.start()
    emitted = 0
    while frame := replay.next_frame():
        print(
            json.dumps(
                {
                    "sequence_index": frame.sequence_index,
                    "source_offset_seconds": frame.source_offset_seconds,
                    "replay_offset_seconds": frame.replay_offset_seconds,
                    "epoch_id": frame.metadata.epoch_id,
                    "label": frame.label.value,
                    "supervised_label_available": frame.supervised_label_available,
                    "stimulus_code": frame.metadata.stimulus_code,
                    "target_probability": frame.target_probability,
                },
                sort_keys=True,
            )
        )
        emitted += 1
        if args.max_frames is not None and emitted >= args.max_frames:
            break


if __name__ == "__main__":
    main()
