# VyTools

### Python toolkit using the Keras AI library for generating Vital wavetable source .wav files

<br>

```diff
+ This program is free and open source software (FOSS)
```

<br>

### Installation

In the root directory run `pip install -r requirements.txt`.

### Running the code

Once the dependencies are installed, run `python main.py <mode string> <optional args>`.

There are 3 different modes for VyTools: `DataAcquisition`, `Training`, or `Generation`.

First, a dataset must be built from your Vital preset files in the `DataAcquisition` step (note that you also need to pass an arg in for your Vital preset directory). wav files will be parsed from your Vital presets and validated (ie, non-unique samples will be deleted to save memory and prevent unnecessary re-training on the same sample).

For the `Training` step, the .wav files in the `\\dataset` folder will be extracted to feature `.bin` files and used to train a Keras model (the trained model will be saved to the `\\models` folder for reuse). You may also pass in a boolean argument to re-use existing saved feature .bin files.

Finally, the `Generation` step will load your saved model and generate a new .wav clip from it.

### License

The MIT License

### About

[Vital](https://vital.audio/) is an open-source wavetable synthesizer similar to Serum.

This project was developed on a Windows machine, some porting may be necessary for Linux or iOS.

### Dependencies

python 3.11
pip  
keras  
numpy  
wave
