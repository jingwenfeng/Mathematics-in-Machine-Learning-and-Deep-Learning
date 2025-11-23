import os
import glob

from midiphrase import csv_to_midi


def main(data_folder: str = "DATA", out_folder: str = "DATA_MIDI") -> None:
    """
    Convert all CSV files in data_folder back to MIDI files in out_folder.
    """
    os.makedirs(out_folder, exist_ok=True)

    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {data_folder}.")

    for csv_path in csv_files:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        midi_out = os.path.join(out_folder, base + ".mid")
        print(f"Converting {csv_path} -> {midi_out}")
        csv_to_midi(csv_path, midi_out_path=midi_out)

    print("Done.")


if __name__ == "__main__":
    main()
