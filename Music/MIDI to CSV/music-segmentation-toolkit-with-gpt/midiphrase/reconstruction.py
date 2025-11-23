from typing import List, Dict, Any
import csv
from collections import defaultdict

import pretty_midi


def csv_to_midi(
    csv_path: str,
    midi_out_path: str = "reconstructed.mid",
    program: int = 0,
) -> None:
    """
    Rebuild a monophonic MIDI file from a CSV created by build_rows_from_phrases.

    The CSV must contain columns:
      phrase_id, note_index, pitch, start_time, end_time, velocity

    phrase_id is currently not used to add gaps, but is preserved
    in case you want to handle phrases separately later.
    """
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    phrases: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phrase_id = int(row["phrase_id"])
        phrases[phrase_id].append(row)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    all_phrase_ids = sorted(phrases.keys())

    for phrase_id in all_phrase_ids:
        notes_in_phrase = phrases[phrase_id]
        notes_in_phrase.sort(key=lambda r: int(r["note_index"]))

        for row in notes_in_phrase:
            pitch = int(row["pitch"])
            start_time = float(row["start_time"])
            end_time = float(row["end_time"])
            velocity = int(row["velocity"])

            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time,
            )
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(midi_out_path)
