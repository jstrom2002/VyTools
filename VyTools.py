## shared.py -- written by JHS 3/15/2024
##
## Shared consts and other values for
## AI generation ecosystem.
## ####################################

import base64
import json
import os
from npy_append_array import NpyAppendArray
import numpy as np
import scipy
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import wave

'''
    HOW TO USE:

    fill_dataset(vital_path)
    generate_model('')
    generate_wav('')
'''



# Constants
# =========
dataset_folder = os.curdir + '\\dataset'
print_during_training: bool = True # set to false for faster training.
training_file = os.curdir + '\\features\\data.npy'
labels_file = os.curdir + '\\features\\labels.npy'
model_file = os.curdir + '\\models\\model.h5'
vital_path = "C:\\Users\\jstro\\OneDrive\\Documents\\Vital" # could replace with sys.argv[1]        



# Custom mean squared error loss function
def custom_mse_loss(y_true, y_pred) -> np.array:
    return np.mean(np.square(y_true - y_pred))

# Function to save generated audio as a .wav file
def save_audio_wav(audio_data, file_path, sample_rate=44100):
    scaled_data = np.int16(audio_data * 32767)
    scipy.io.wavfile.write(file_path, sample_rate, scaled_data)

def load_audio(wave_file) -> dict:
    ''''
        Load and preprocess audio data.
    '''

    ifile = wave.open(wave_file)
    channels = ifile.getnframes()

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(ifile.readframes(channels), dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0 by dividing 
    # by the max int16 val.                                                      
    audio_normalised = audio_as_np_float32 / (np.iinfo(np.int16).max)   
    if channels == 2:
        # audio_stereo = np.empty((int(len(audio_normalised)/channels), 2))
        # audio_stereo[:,0] = audio_normalised[range(0,len(audio_normalised),2)]
        # audio_stereo[:,1] = audio_normalised[range(1,len(audio_normalised),2)]

        # can't currently handle stereo data, skip this type.
        return {}

    return {
        'channels': ifile.getnchannels(),
        'samples': ifile.getnframes(),
        'data': audio_normalised
    }

def generate_model(dataset_folder: str, feature_filename: str, labels_filename: str, output_filename: str, use_existing_files: bool=False) -> None:
    """
        Generate numpy .npy binary feature/labels files from all audio in the dataset.
        The file produced will have x and y values where x is float32 audio data and y
        is an integer index where this file occurred in the wav data. Then, generate a
        trained .keras/.h5 model and output into our local '\models' folder.
    
        Args:
            dataset_folder (str): folder with all training dataset files.
            feature_filename (str): name of the output .npy data file.
            labels_filename (str): name of the output .npy labels file.
            output_filename (str): name of the output numpy file.
 
        Returns:
            str with name of written model file if successful, else null string.    
    """

    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))

    if not use_existing_files:
        open(feature_filename,'wb')
        open(labels_filename,'wb')

        # Reset training files by default.
        npaa = open(feature_filename,'ab')
        labels = open(labels_filename,'ab')
        
        # Open files and stream float32/int32 data into them continuously.
        for root, _, files in os.walk(dataset_folder):
            for filename in files:    
                if filename.endswith('.wav'):
                    abs_filepath = os.path.join(root,filename)
                    ifile = load_audio(abs_filepath) 
                    if 'data' in ifile.keys():
                        for x in ifile['data']:
                            npaa.write(x)
                        for x in np.arange(len(ifile['data']), dtype=np.int32):
                            labels.write(x)
        npaa.close()
        labels.close()

    # Load dataset.
    x_train = np.array([], np.float32)
    with open(feature_filename, 'rb') as f:
        while 1:
            byte_s: np.float32 = f.read(4)
            if not byte_s:
                break
            x_train.append(byte_s)

    y_train = np.array([], np.float32)
    with open(labels_file, 'rb') as f:
        while 1:
            byte_s: np.float32 = f.read(4)
            if not byte_s:
                break
            x_train.append(byte_s)

    # Build and compile the generator model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=x_train.shape[1:]),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse')
    #model.compile(optimizer='adam', loss=custom_mse_loss) # use custom MSE function

    # Train the generator
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Export generated model.
    model_filename = os.curdir + "\\model\\" + output_filename
    model.save(model_filename)

    return model_filename


# Function to generate audio using the trained model
def generate_audio(model, seed_input, sequence_length):
    generated_audio = []
    current_sequence = seed_input

    # Generate audio sequence step by step
    for _ in range(sequence_length):
        # Predict the next timestep
        next_timestep = model.predict(current_sequence[np.newaxis, :, :])
        
        # Append the predicted timestep to the generated audio
        generated_audio.append(next_timestep[0, -1, 0])
        
        # Update the current sequence by removing the first timestep and appending the predicted timestep
        current_sequence = np.concatenate([current_sequence[:, 1:, :], next_timestep], axis=1)

    return np.array(generated_audio)

def generate_wav(model_filepath: str, output_filename: str) -> str:
    """
        Load trained .keras model and use it to generate new .wav content.

        Args:
            model_filename (str): name of the model .keras model file.
            output_filename (str): name of the output generated file.
 
        Returns:
            str with name of written generated file if successful, else null string.
    """

    # Load the trained generator model
    generator = load_model(model_filepath)  # Update with your model file path
    
    # Seed input for generation
    seed_input = np.random.random((1, 10, 1))  # Example seed input, adjust as needed
    
    # Generate audio
    sequence_length = 44100  # Adjust as needed for the desired length of audio    
    generated_audio = generate_audio(generator, seed_input, sequence_length)

    # Generate.
    open(output_filename, 'w').write(generated_audio)

    return output_filename


def save_base64_str_to_wav(b64str: str, output_filename: str) -> None:
    data = base64.b64decode(b64str)
    with wave.open(output_filename, "wb") as out_f:
        out_f.setnchannels(1)
        out_f.setsampwidth(2) # number of bytes
        out_f.setframerate(44100)
        out_f.writeframesraw(data)
        out_f.close()


def fill_dataset(vital_root_doc_folder: str) -> None:
    '''
        Recurse vital root folder into .wav files in '\dataset' folder.

        Args:
            vital_root_doc_folder (str): folder for the Vital synth that contains all .vital preset files.
 
        Returns:
            None.  
    '''
    extracted_count = len([name for name in os.listdir(dataset_folder) if os.path.isfile(name)]) + 1
    for root, _, files in os.walk(vital_root_doc_folder):
        for filename in files:    
            if os.path.splitext(filename)[1] == '.vital':
                abs_filepath = os.path.join(root,filename)
                vital_file = json.loads(open(abs_filepath, 'r').read())
                if isinstance(vital_file,dict):
                    try:
                        if 'settings' in vital_file.keys():
                            for wt in vital_file['settings']['wavetables']:
                                for g in wt['groups']:
                                    for cmp in g['components']:
                                        if 'audio_file' in cmp.keys() and cmp['audio_file'] and len(cmp['audio_file']) > 0:
                                            save_base64_str_to_wav(cmp['audio_file'], dataset_folder + 
                                                                                '\\extracted' + str(extracted_count) + '.wav')
                                            extracted_count = extracted_count + 1
                                        if 'keyframes' in cmp.keys() and cmp['keyframes'] and len(cmp['keyframes']) > 0:
                                            for kf in cmp['keyframes']:
                                                if 'wave_data' in kf.keys():
                                                    save_base64_str_to_wav(kf['wave_data'], dataset_folder + 
                                                                                        '\\extracted' + str(extracted_count) + '.wav')
                                                    extracted_count = extracted_count + 1
                    except Exception as e:
                        print(str(e))                    
    



if __name__ == "__main__":
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)    

    # # 1. Fill dataset.
    #fill_dataset(vital_path)

    # 2. Train dataset.
    model_file = generate_model(dataset_folder, training_file, labels_file, model_file, use_existing_files=False)

    # 3. Run trained model and generate file.
    generate_wav(model_file,'\\output\\generated.wav')