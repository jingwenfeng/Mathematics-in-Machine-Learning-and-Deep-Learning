import os
import glob

from midiphrase import (
    extract_monophonic_notes,
    chunk_notes,
    call_gpt_for_boundaries_chunk,
    build_phrases_from_indices,
    build_rows_from_phrases,
    save_rows_to_csv,
    set_openai_client,
)


def main(midi_folder: str = "MIDI", output_folder: str = "DATA") -> None:
    """
    Convert all MIDI files in midi_folder to phrase level CSV files in output_folder.
    """
    set_openai_client()

    os.makedirs(output_folder, exist_ok=True)

    midi_files = glob.glob(os.path.join(midi_folder, "*.mid"))
    midi_files += glob.glob(os.path.join(midi_folder, "*.midi"))

    print(f"Found {len(midi_files)} MIDI files in {midi_folder}.")

    for midi_path in midi_files:
        print(f"\nProcessing {midi_path}...")
        note_events = extract_monophonic_notes(midi_path)

        all_phrase_groups = []
        chunks = chunk_notes(note_events, chunk_size=300)

        for chunk in chunks:
            if not chunk:
                continue
            local_phrases = call_gpt_for_boundaries_chunk(chunk)
            chunk_offset = chunk[0]["index"]
            for phrase in local_phrases:
                local_indices = phrase.get("indices", [])
                global_indices = [chunk_offset + idx for idx in local_indices]
                all_phrase_groups.append({"indices": global_indices})

        phrases = build_phrases_from_indices(note_events, all_phrase_groups)
        print(f"Total phrases from GPT: {len(phrases)}")

        rows = build_rows_from_phrases(phrases)

        base_name = os.path.splitext(os.path.basename(midi_path))[0]
        csv_path = os.path.join(output_folder, base_name + ".csv")
        save_rows_to_csv(rows, csv_path)
        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
