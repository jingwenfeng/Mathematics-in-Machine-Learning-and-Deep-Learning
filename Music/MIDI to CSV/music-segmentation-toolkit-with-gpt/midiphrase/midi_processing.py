from typing import List, Dict, Any
import pretty_midi


def midi_to_note_name(pitch: int) -> str:
    """
    Convert MIDI pitch number to human readable note name, for example:
    60 -> C4, 61 -> C#4, 62 -> D4.
    """
    names = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = pitch // 12 - 1
    return f"{names[pitch % 12]}{octave}"


def extract_monophonic_notes(midi_path: str) -> List[Dict[str, Any]]:
    """
    Load a MIDI file and extract notes as a monophonic sequence.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempo = pm.estimate_tempo()  # bpm

    # choose instrument with most notes as melody
    if not pm.instruments:
        raise ValueError("No instruments in MIDI file.")
    inst = max(pm.instruments, key=lambda ins: len(ins.notes))

    notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))

    # simple monophonic conversion: keep highest pitch at same start, remove overlaps
    mono = []
    same_time_tol = 1e-3
    i = 0
    while i < len(notes):
        t = notes[i].start
        simultaneous = []
        while i < len(notes) and abs(notes[i].start - t) < same_time_tol:
            simultaneous.append(notes[i])
            i += 1
        chosen = max(simultaneous, key=lambda n: n.pitch)

        if mono and chosen.start < mono[-1].end:
            new_start = mono[-1].end
            new_end = max(new_start, chosen.end)
            if new_end <= new_start:
                continue
            chosen = pretty_midi.Note(
                velocity=chosen.velocity,
                pitch=chosen.pitch,
                start=new_start,
                end=new_end
            )
        mono.append(chosen)

    final_notes = []
    for n in mono:
        if final_notes and n.start < final_notes[-1].end:
            continue
        final_notes.append(n)

    def sec_to_beat(t):
        return t * tempo / 60.0

    note_events = []
    for idx, n in enumerate(final_notes):
        start_sec = float(n.start)
        end_sec = float(n.end)
        start_beat = sec_to_beat(start_sec)
        end_beat = sec_to_beat(end_sec)
        note_events.append({
            "index": idx,
            "pitch": int(n.pitch),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_beat": start_beat,
            "end_beat": end_beat,
        })

    return note_events
