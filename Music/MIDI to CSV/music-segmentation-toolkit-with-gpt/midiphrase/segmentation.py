from typing import List, Dict, Any, Optional
import os
import json
import csv

from openai import OpenAI

_client: Optional[OpenAI] = None


def set_openai_client(api_key: Optional[str] = None) -> None:
    """
    Configure the global OpenAI client for this library.

    If api_key is None, use the OPENAI_API_KEY environment variable.
    You must call this once before using call_gpt_for_boundaries_chunk.
    """
    global _client
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set and no api_key was passed to set_openai_client."
            )
    _client = OpenAI(api_key=api_key)


SYSTEM_PROMPT_SEGMENT = """
You are a music theory expert and phrase segmentation assistant.

You are given a strictly monophonic melody as a JSON array of notes.
Each note has fields:
  - index: integer
  - pitch: MIDI pitch number
  - note_name: string like "C4"
  - start_time: float, seconds
  - end_time: float, seconds
  - duration: float, seconds
  - velocity: integer

Your task:
1. Segment this melody into phrases.
2. Each phrase is a list of note indices that form a coherent musical unit.
3. Phrases must:
   - Preserve original order.
   - Use each note index at most once.
   - Not overlap.
4. Use musical intuition:
   - Cadence points.
   - Longer gaps between notes.
   - Repetitive motifs.
   - Breath or bow like groupings.

Return JSON with this structure:

{
  "phrases": [
    { "indices": [0, 1, 2, 3] },
    { "indices": [4, 5, 6] },
    ...
  ]
}

Do not include any additional fields. Do not include comments or text outside of valid JSON.
"""


def chunk_notes(note_events: List[Dict[str, Any]], chunk_size: int = 300) -> List[List[Dict[str, Any]]]:
    """
    Split a long list of notes into smaller chunks for GPT.
    """
    chunks: List[List[Dict[str, Any]]] = []
    for i in range(0, len(note_events), chunk_size):
        chunks.append(note_events[i:i + chunk_size])
    return chunks


def call_gpt_for_boundaries_chunk(note_chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Call GPT on a single chunk of notes and return phrase index groups.

    Returns a list like:
      [
        {"indices": [0, 1, 2]},
        {"indices": [3, 4]}
      ]

    Note indices are local to the chunk. You must offset them when
    combining multiple chunks, which is done in higher level code.
    """
    if _client is None:
        raise RuntimeError("OpenAI client is not set. Call set_openai_client() first.")

    notes_json = json.dumps(note_chunk, ensure_ascii=False)

    user_prompt = (
        "Here is a strictly monophonic melody as a JSON array called \"notes\".\n"
        "Apply the rules from the system message to segment it into phrases.\n"
        "Return JSON with a 'phrases' array as specified.\n\n"
        "notes = "
        + notes_json
    )

    response = _client.chat.completions.create(
        model="gpt-5.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SEGMENT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    phrases = data.get("phrases", [])
    if not isinstance(phrases, list):
        raise ValueError("Model response does not contain a valid 'phrases' list.")
    return phrases


def build_phrases_from_indices(
    all_notes: List[Dict[str, Any]],
    all_phrase_index_groups: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """
    Convert phrase index groups into lists of note dictionaries.

    all_phrase_index_groups is expected to be a list of:
      {"indices": [0, 1, 2, ...]}
    where the indices refer to positions in all_notes.
    """
    index_to_note = {note["index"]: note for note in all_notes}
    phrases: List[List[Dict[str, Any]]] = []

    for group in all_phrase_index_groups:
        indices = group.get("indices", [])
        phrase_notes: List[Dict[str, Any]] = []
        for idx in indices:
            if idx in index_to_note:
                phrase_notes.append(index_to_note[idx])
        phrase_notes.sort(key=lambda n: n["index"])
        if phrase_notes:
            phrases.append(phrase_notes)

    return phrases


def build_rows_from_phrases(phrases: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Convert phrase note lists into CSV rows.

    Each row has:
      phrase_id, note_index, pitch, note_name, start_time, end_time, duration, velocity
    """
    rows: List[Dict[str, Any]] = []
    for phrase_id, phrase_notes in enumerate(phrases):
        for note in phrase_notes:
            row = {
                "phrase_id": phrase_id,
                "note_index": note["index"],
                "pitch": note["pitch"],
                "note_name": note["note_name"],
                "start_time": note["start_time"],
                "end_time": note["end_time"],
                "duration": note["duration"],
                "velocity": note["velocity"],
            }
            rows.append(row)
    return rows


def save_rows_to_csv(rows: List[Dict[str, Any]], csv_path: str) -> None:
    """
    Save rows returned by build_rows_from_phrases to a CSV file.
    """
    if not rows:
        raise ValueError("No rows to save to CSV.")

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
