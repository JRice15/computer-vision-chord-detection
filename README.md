# MagiChord

### Data Pipeline
* collect_data.py:
    * collect video data and corresponding chord annotations
    * this saves .npy data of chords to the `data/` directory, and raw video to `raw_data/`
    * you have to manually sort data into one of the following:
        * `data/image_model_train`: data for training the single-image model
        * `data/inference_model_train`: test set for image model, train set for inference model
        * `data/inference_model_test`: test for inference model
* process_data.py:
    * this runs `src/detect_fretboard.py` on the `raw_data`, and saving the fretboard videos to `data`. It skips data that has already been run, unless you use the `--overwrite` flag
* train_image_model.py:
    * trains model to learn fret placements from individual frames of video. This model is pretty big, and should be run on a GPU
    * this model uses the `model_config.json` file
* test_image_model.py (optional):
    * evaluate the performance of the image model on the `data/inference_model_train` set
    * this is run by train_image_model at the end automatically
* train_inference_model.py:
    * train the inference model on the data in `data/inference_model_train`
* test_inference_model.py:
    * test the inference model (and image model) on the test set in `data/inference_model_test`
