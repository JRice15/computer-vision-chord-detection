# MagiChord

### Data Pipeline
* collect_data.py:
    * collect video data and corresponding chord annotations
    * this saves .npy data of chords to the `data/` directory, and raw video to `raw_data/`
* process_data.py:
    * this runs `detect_fretboard.py` on the `raw_data`, and saving the fretboard videos to `data`. It skips data that has already been run, unless you use the
    `--overwrite` flag
* train_image_model.py:
    * trains model to learn fret placements from individual frames of video. This model is pretty big, and should be run on a GPU
* test_image_model.py (optional):
    * evaluate 