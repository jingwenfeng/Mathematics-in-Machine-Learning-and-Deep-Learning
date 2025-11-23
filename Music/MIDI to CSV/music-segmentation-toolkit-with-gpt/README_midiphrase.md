# midiphrase

`midiphrase` is a small Python library and set of command line tools for:

1. Reading **MIDI** files and extracting a **monophonic** note sequence.  
2. Using **OpenAI GPT** to segment the melody into **musical phrases**.  
3. Saving phrases as labeled rows in **CSV** files.  
4. Reconstructing MIDI from those CSV files.

It is designed to be:

- Reproducible: your processing lives in a Python package, not only in a notebook.  
- Scriptable: you can call it from the command line or from Python.  
- Extensible: you can swap the model, change prompts, or plug in your own segmentation logic.

---

## 1. Requirements

- **Python**: 3.10 or newer is recommended.  
- **Operating system**: Linux, macOS, or Windows.

### Core Python dependencies

These are installed automatically when you install the project:

- `openai` – for calling GPT models  
- `pretty_midi` – for reading and writing MIDI files  

Optional but useful for your workflow:

- `python-dotenv` – load `OPENAI_API_KEY` from a `.env` file  
- `jupyter` – if you still want to use notebooks

---

## 2. Project structure

A typical `midiphrase` project layout looks like this:

```text
your-project/
  README.md
  pyproject.toml
  requirements.txt          # optional, for your dev environment
  main_rhythm.ipynb         # your original notebook (optional, for reference)
  midiphrase/
    __init__.py
    midi_processing.py
    segmentation.py
    reconstruction.py
  scripts/
    midi_to_csv.py
    csv_to_midi.py
```

### What each part does

- `midiphrase/`  
  Core library code:
  - `midi_processing.py` – read MIDI and turn it into a list of note dictionaries  
  - `segmentation.py` – GPT prompt, chunking, phrase index handling, CSV writing  
  - `reconstruction.py` – take CSV and rebuild MIDI  
  - `__init__.py` – public API exports

- `scripts/`  
  Convenience command line entry points:
  - `midi_to_csv.py` – batch process MIDI → CSV (using GPT segmentation)  
  - `csv_to_midi.py` – batch process CSV → MIDI

- `pyproject.toml`  
  Defines the package metadata and dependencies so you can install with `pip`.

- `requirements.txt`  
  Optional file that pins dependencies for your **development environment**.  
  The packaging info still lives in `pyproject.toml`.

- `README.md`  
  This file – explains what the project does and how to use it.

---

## 3. Installation

You typically want a **virtual environment** for this project.

### 3.1 Clone / copy the project

If this lives in a git repo, clone it; otherwise just put all files into one folder:

```bash
git clone <your-repo-url> midiphrase-project
cd midiphrase-project
```

Replace `<your-repo-url>` with your actual repository URL if you use git.

### 3.2 Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

- macOS / Linux:

  ```bash
  source .venv/bin/activate
  ```

- Windows (PowerShell):

  ```powershell
  .venv\Scripts\Activate
  ```

You should now see something like `(.venv)` at the start of your terminal prompt.

### 3.3 Install in editable mode

This makes your package importable as `midiphrase` and keeps it linked to your source files.

```bash
pip install --upgrade pip
pip install -e .
```

What this does:

- Reads `pyproject.toml`  
- Installs `midiphrase` in **editable** mode  
- Installs the dependencies:
  - `openai`
  - `pretty_midi`

If you also want `python-dotenv` for `.env` support, either:

- Install it directly:

  ```bash
  pip install python-dotenv
  ```

- Or add it to `requirements.txt` and run:

  ```bash
  pip install -r requirements.txt
  ```

> **Note:**  
> The **official dependency list for packaging** is in **`pyproject.toml`**.  
> `requirements.txt` is just for your local development environment and is not required for users who install from PyPI.

---

## 4. Setting your OpenAI API key

The library expects the OpenAI API key to be available as an environment variable `OPENAI_API_KEY`.

### 4.1 Directly in the shell

- macOS / Linux:

  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  ```

- Windows (PowerShell):

  ```powershell
  $env:OPENAI_API_KEY="your_api_key_here"
  ```

You can verify:

```bash
echo $OPENAI_API_KEY         # macOS / Linux
echo $env:OPENAI_API_KEY     # Windows PowerShell
```

### 4.2 Using `.env` and `python-dotenv` (optional)

If you prefer not to expose your key in the shell:

1. Create a file named `.env` in your project root:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

2. In **your own scripts or notebooks**, load it like:

   ```python
   from dotenv import load_dotenv
   load_dotenv()  # reads the .env file
   ```

3. After that, `midiphrase` will find `OPENAI_API_KEY` from the environment.

> Inside the **library**, we do **not** hard-depend on `python-dotenv`.  
> It is your choice whether to use it at the application level.

---

## 5. Command line usage

Installing the project with `pip install -e .` also installs two console scripts:

- `midiphrase-midi-to-csv`  
- `midiphrase-csv-to-midi`

### 5.1 Prepare folders

By default the scripts expect:

- A folder named `MIDI/` containing `.mid` or `.midi` files  
- A folder named `DATA/` for CSV output  
- A folder named `DATA_MIDI/` for reconstructed MIDI output

You can create those folders and put your MIDI files into `MIDI/`:

```bash
mkdir -p MIDI DATA DATA_MIDI
cp path/to/your/*.mid MIDI/
```

(Or move the MIDI files in any way you prefer.)

### 5.2 MIDI → CSV (with GPT phrase segmentation)

```bash
midiphrase-midi-to-csv
```

This will:

1. Read all `.mid` and `.midi` files in `./MIDI`  
2. Extract monophonic note sequences  
3. Break them into chunks (default 300 notes per chunk)  
4. Call the OpenAI API for each chunk to get phrase boundaries  
5. Combine chunk results into global phrase indices  
6. Save one CSV per MIDI file in `./DATA`

You will see output like:

```text
Found 3 MIDI files in MIDI.

Processing MIDI/example1.mid...
Total phrases from GPT: 18
Saved CSV to DATA/example1.csv

Processing MIDI/example2.mid...
...
```

> **Note:** This step uses the OpenAI API and will incur latency and cost depending on the length and number of files.

### 5.3 CSV → MIDI (reconstruction)

```bash
midiphrase-csv-to-mиди
```

This will:

1. Read all `.csv` files in `./DATA`  
2. Reconstruct a monophonic MIDI file from each CSV  
3. Save output `.mid` files into `./DATA_MIDI`

You will see output like:

```text
Found 3 CSV files in DATA.
Converting DATA/example1.csv -> DATA_MIDI/example1.mid
Converting DATA/example2.csv -> DATA_MIDI/example2.mid
Done.
```

You can then open `DATA_MIDI/example1.mid` in any MIDI player or DAW.

---

## 6. Using midiphrase as a Python library

You can also use `midiphrase` directly in your own Python code or notebooks.

### 6.1 Basic example: process a single MIDI file

```python
from midiphrase import (
    set_openai_client,
    extract_monophonic_notes,
    chunk_notes,
    call_gpt_for_boundaries_chunk,
    build_phrases_from_indices,
    build_rows_from_phrases,
    save_rows_to_csv,
    csv_to_midi,
)

# This reads the OPENAI_API_KEY from the environment
set_openai_client()

# 1. Extract monophonic note sequence from MIDI
note_events = extract_monophonic_notes("example.mid")

# 2. Split into chunks for GPT
chunks = chunk_notes(note_events, chunk_size=300)

all_phrase_groups = []

for chunk in chunks:
    if not chunk:
        continue
    # call GPT on this chunk
    local_phrases = call_gpt_for_boundaries_chunk(chunk)

    # indices in local_phrases are relative to this chunk,
    # so we convert them to global indices
    chunk_offset = chunk[0]["index"]
    for phrase in local_phrases:
        local_indices = phrase.get("indices", [])
        global_indices = [chunk_offset + idx for idx in local_indices]
        all_phrase_groups.append({"indices": global_indices})

# 3. Convert index groups into phrase note lists
phrases = build_phrases_from_indices(note_events, all_phrase_groups)

# 4. Convert phrase notes into flat rows
rows = build_rows_from_phrases(phrases)

# 5. Save rows to CSV
save_rows_to_csv(rows, "example.csv")

# 6. Optionally, reconstruct MIDI from CSV
csv_to_midi("example.csv", midi_out_path="example_reconstructed.mid")
```

This is basically what the CLI script does, but you can customize every step.

---

## 7. Library API overview

Here is a summary of the main functions exposed in `midiphrase.__init__.py`.

### 7.1 `set_openai_client(api_key: Optional[str] = None) -> None`

- Configures the global OpenAI client used by segmentation.  
- If `api_key` is omitted, reads from `OPENAI_API_KEY` environment variable.  
- Must be called once before using `call_gpt_for_boundaries_chunk`.

Example:

```python
from midiphrase import set_openai_client

set_openai_client()                 # uses env var
# or
set_openai_client("sk-...your-key")
```

---

### 7.2 MIDI processing

```python
from midiphrase import midi_to_note_name, extract_monophonic_notes
```

#### `midi_to_note_name(pitch: int) -> str`

- Converts a MIDI pitch (e.g. `60`) to a note string (e.g. `"C4"`).

#### `extract_monophonic_notes(midi_path: str) -> List[dict]`

- Reads the first instrument in the MIDI file.  
- Sorts notes by `(start_time, pitch)`.  
- Filters to a **monophonic** sequence by dropping overlapping notes.  
- Returns a list of dicts, each like:

  ```python
  {
      "index": 0,
      "pitch": 60,
      "note_name": "C4",
      "start_time": 0.0,
      "end_time": 0.5,
      "duration": 0.5,
      "velocity": 80,
  }
  ```

---

### 7.3 Segmentation (GPT based)

```python
from midiphrase import (
    SYSTEM_PROMPT_SEGMENT,
    chunk_notes,
    call_gpt_for_boundaries_chunk,
    build_phrases_from_indices,
    build_rows_from_phrases,
    save_rows_to_csv,
)
```

#### `SYSTEM_PROMPT_SEGMENT: str`

- The default system prompt used to instruct GPT how to segment melodies into phrases.  
- You can import and modify it in your own code if you want to experiment.

#### `chunk_notes(note_events: List[dict], chunk_size: int = 300) -> List[List[dict]]`

- Splits a long list of notes into smaller chunks.  
- Default chunk size is 300 notes.  
- Returns a list of chunks, each a list of note dicts.

#### `call_gpt_for_boundaries_chunk(note_chunk: List[dict]) -> List[dict]`

- Calls the OpenAI Chat Completions API on one chunk.  
- Expects `set_openai_client` to have been called beforehand.  
- Returns a list of phrase descriptors, each like:

  ```python
  {"indices": [0, 1, 2, 3]}
  ```

- These indices are **local** to the chunk and must be shifted to global indices when combining chunks.

#### `build_phrases_from_indices(all_notes: List[dict], all_phrase_index_groups: List[dict]) -> List[List[dict]]`

- Takes the global note list and a list of index groups like:

  ```python
  [{"indices": [0, 1, 2]}, {"indices": [3, 4]}]
  ```

- Returns a list of phrases, where each phrase is a list of note dicts.

#### `build_rows_from_phrases(phrases: List[List[dict]]) -> List[dict]`

- Turns phrase note lists into **flat rows** suitable for CSV.  
- Each row has keys:

  - `phrase_id`  
  - `note_index`  
  - `pitch`  
  - `note_name`  
  - `start_time`  
  - `end_time`  
  - `duration`  
  - `velocity`

#### `save_rows_to_csv(rows: List[dict], csv_path: str) -> None`

- Writes the rows to a CSV file at `csv_path`.  
- Uses the keys from the first row as CSV headers.

---

### 7.4 Reconstruction

```python
from midiphrase import csv_to_midi
```

#### `csv_to_midi(csv_path: str, midi_out_path: str = "reconstructed.mid", program: int = 0) -> None`

- Reads a CSV produced by `save_rows_to_csv`.  
- Groups rows by `phrase_id` (for future phrase-wise processing).  
- Writes a monophonic MIDI to `midi_out_path`.  
- Uses `program` as the MIDI instrument program number (0 is usually Acoustic Grand Piano).

---

## 8. Configuration details

There are a few things you might want to tweak.

### 8.1 Model name

In the current code, `call_gpt_for_boundaries_chunk` uses:

```python
model="gpt-5.1-mini"
```

You can change this to any compatible model you have access to, for example:

```python
response = _client.chat.completions.create(
    model="gpt-4.1-mini",
    ...
)
```

Just make sure the model supports `response_format={"type": "json_object"}`.

### 8.2 Chunk size

The `chunk_size` in `chunk_notes()` controls how many notes go to GPT at once:

- Larger chunks = better global context but more tokens (more cost, more latency)  
- Smaller chunks = cheaper and faster but may miss longer phrase structures

You can call:

```python
chunks = chunk_notes(note_events, chunk_size=200)
```

or override the default in your own code.

### 8.3 Monophonic assumption

The current `extract_monophonic_notes`:

- Uses only the **first instrument**  
- Enforces **monophony** by discarding overlapping notes  

If you need full polyphony or multi instrument handling, you can:

- Modify `midi_processing.py` to:
  - Merge or separate instruments  
  - Preserve overlapping notes  
  - Add more metadata (e.g., channel, instrument index)

---

## 9. Development notes

If you are developing `midiphrase` itself (not just using it):

### 9.1 Installing dev dependencies

If you maintain a `requirements.txt`, you can include things like:

```text
openai>=1.0.0
pretty_midi>=0.2.10
python-dotenv>=1.0.0
pytest>=8.0.0
```

Then install with:

```bash
pip install -r requirements.txt
pip install -e .
```

### 9.2 Running tests

If you add a `tests/` directory and use `pytest`, you can run:

```bash
pytest
```

### 9.3 Editing / extending the library

Common extension points:

- Experiment with new prompts in `SYSTEM_PROMPT_SEGMENT`  
- Try different chunk sizes or note representations  
- Add features to the CSV (e.g. beat positions, key, scale degrees)  
- Add a configuration file instead of hard coding defaults in scripts

---

## 10. FAQ

### Q1. Where should I put the `install` requirements: `requirements.txt` or `pyproject.toml`?

- For **packaging and distribution** (so others can `pip install midiphrase`), the **canonical place is `pyproject.toml`** under `[project].dependencies`.  
- `requirements.txt` is mainly for your **local development environment**:
  - Convenient for `pip install -r requirements.txt`  
  - You can pin exact versions for reproducibility  
- In short:
  - **Users** rely on `pyproject.toml`  
  - **You** might use `requirements.txt` as a convenience

### Q2. Do I still need my original notebook?

You do not **need** it for the library to work, but it can be:

- A useful sandbox to experiment with new ideas  
- A demo you can show or share  

The library is meant to **extract the core logic from the notebook**, so you can keep the notebook light and exploratory.

### Q3. Does this library support polyphonic MIDI?

Currently, the default pipeline assumes **monophonic** lines:

- It picks the first instrument  
- It filters overlapping notes to enforce monophony  

For polyphonic phrase analysis you would need to extend `midi_processing.py` and adjust the segmentation logic.

### Q4. How expensive is segmentation?

It depends on:

- Number of notes  
- Number of files  
- Model you use (`gpt-5.1-mini`, `gpt-4` etc.)  

Each chunk call sends:

- The notes in JSON format  
- The segmentation prompt  

So you should:

- Start with a small test file  
- Monitor your OpenAI usage dashboard  
- Tune `chunk_size` and the number of files accordingly

---

## 11. License

You can choose your own license based on how you want others to use your work.

Example placeholder text (replace with what you actually want):

```text
This project is currently closed source / all rights reserved.
Please do not redistribute without permission.
```

Or use a standard license like MIT, Apache 2.0, or GPL by adding a `LICENSE` file and updating this section.

---

If you follow this README from top to bottom, you should be able to:

1. Install `midiphrase` into a clean environment  
2. Set your `OPENAI_API_KEY`  
3. Convert MIDI → CSV with phrase annotations using GPT  
4. Rebuild MIDI back from CSV  
5. Use the library functions directly in your own code or notebooks
