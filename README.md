# MagiChord

### Data Pipeline
* collect_data.py:
    * collect video data and corresponding chord annotations
    * this saves .npy data of chords to the `data/` directory, and raw video to `raw_data/`
* process_data.py:
    * this runs `detect_fretboard.py` on the `raw_data`, and saving the fretboard videos to `data`. It skips data that has already been run, unless you use the
    `--overwrite` flag

