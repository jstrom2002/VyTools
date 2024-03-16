## VyTools.py -- written by JHS 3/15/2024
##
## Main entrypoint for populating a dataset
## from the installed Vital root directory,
## training a model with it, and generating
## a single output .wav file using this
## model.
## ########################################

import base64
import filecmp
import json
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import math
import os
import numpy as np
import struct
import sys
import time
import wave



# Constants
# =========
dataset_folder = os.curdir + '\\dataset'
print_during_training: bool = True # set to false for faster training.
training_file = os.curdir + '\\features\\data.bin'
details_file = os.curdir + '\\features\\details.txt'
model_file = os.curdir + '\\models\\model.h5'
dataset_sample_size = 2048 # all data smaller than this size will be upscaled by repeating the smaller size data until it fits.
default_sample_rate = 44100


def load_audio(wave_file: str) -> dict:
    ''''
        Load and preprocess audio data.

        Args:
            wave_file (str): the filepath to a .wav formatted audio file.

        Returns:
            A dict value with entries comparable to those returned by 'wave.open()'.
    '''

    ifile = wave.open(wave_file)
    channels = ifile.getnframes()
    audio_as_np_int16 = None
    if channels == 2: # handle stereo .wav files by converting to mono. TO DO: test
        ifile.split_to_mono()
        dataArr = ifile.readframes(channels)
        print(dataArr)
        assert len(dataArr[0]) == len(dataArr[1])
        dataArr = dataArr[0] * 0.5 + dataArr[1] * 0.5
        audio_as_np_int16 = np.frombuffer(dataArr, dtype=np.int16)
    else:
        audio_as_np_int16 = np.frombuffer(ifile.readframes(channels), dtype=np.int16)

    # Convert buffer to float32 using NumPy. Then normalize to [-1,1] by dividing by the abs(max)                                                                                
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
    max_val = np.max(np.abs(audio_as_np_float32), axis=0)
    if max_val == 0:
        max_val = 0.00001
    audio_as_np_float32 /= max_val

    # Normalise float32 array so that values are between -1.0 and +1.0 by dividing 
    # by the max int16 val.                                                      
    audio_normalized = audio_as_np_float32 / (np.iinfo(np.int16).max)   

    # Keep only the same small number of N samples of each wav sample. If an audio
    # file contains too few samples, repeat the original sample N times until it's
    # large enough.
    while len(audio_normalized) < dataset_sample_size:
        audio_normalized = np.concatenate([audio_normalized, audio_normalized])
    audio_normalized = audio_normalized[:dataset_sample_size]

    assert(len(audio_normalized) >= dataset_sample_size)

    return {
        'channels': ifile.getnchannels(),
        'samples': ifile.getnframes(),
        'data': audio_normalized
    }

def extract_features(dataset_folder: str, feature_filename: str, details_filename: str) -> None:
    '''
        Extract all features from the collected dataset and write to .bin files with all the data stored
        in contiguous bytes as stored from float32 or int32 float data or int indices.

        Args:
            dataset_folder (str): location of the dataset folder with all .wav files for extraction.
            feature_filename (str): output location of the features.bin file.
            details_filename (str): output locatino of the details.txt file.

        Return:
            None.
    '''

    open(feature_filename,'wb')

    # Reset training files by default.
    features = open(feature_filename,'ab')
        
    # Open files and stream float32/int32 data into them continuously.
    file_count = 0
    float32_count = 0
    for root, _, files in os.walk(dataset_folder):
        for filename in files:    
            if filename.endswith('.wav'):
                abs_filepath = os.path.join(root,filename)
                ifile = load_audio(abs_filepath) 
                if 'data' in ifile.keys():
                    file_count = file_count + 1
                    float32_count = float32_count + len(ifile['data'])
                    for x in ifile['data']:
                        features.write(x)
    features.close()


    # Finally, include a details.txt file with some useful dataset details.
    details = open(details_filename,'w')
    details.write(str({
        'dataset_sample_size': dataset_sample_size,
        'feature_filename': feature_filename,
        'float32_count': float32_count,
        'file_count': file_count,
    }))
    details.close()


def generate_model(feature_filename: str, output_filename: str) -> None:
    """
        Generate a trained .keras/.h5 model and output into our local '\models' folder.
    
        Args:
            feature_filename (str): name of the output .npy data file.
            output_filename (str): name of the output numpy file.
 
        Returns:
            str with name of written model file if successful, else null string.    
    """

    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))

    # Load dataset from binary files.
    x_train: list[np.float32] = []
    with open(feature_filename, 'rb') as f:
        while 1:
            byte_s = f.read(4)
            if not byte_s:
                break
            float_data: np.float32 = struct.unpack('f', byte_s)
            x_train.append(float_data)

    # Force numpy to split this array into N subarrays of size=dataset_sample_size
    final_arr = np.array(x_train,np.float32)
    final_arr = final_arr.reshape(dataset_sample_size, math.floor(len(x_train)/dataset_sample_size))
    x_train.clear()
    print('final_arr shape: ' + str(final_arr.shape))

    # Build and compile the generator model.
    # NOTE: for keras, a dataset with 30 images, 50x50 pixels and 3 channels, will require use of "input_shape=(50,50,3)"
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(final_arr.shape)),
        Dense(1, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse')


    # Train the generator
    model.fit(final_arr, epochs=10, batch_size=dataset_sample_size)

    # Export generated model.
    model_filename = os.curdir + "\\model\\" + output_filename
    model.save(model_filename)

    return model_filename


# Function to generate audio using the trained model
def generate_audio(model, seed_input, sequence_length: int) -> np.array:
    ''' 
        A function that will output a numpy array of generated audio.

        Args:
            model (any): a Keras model.
            seed_input (any): an array of seed data to randomize the Keras model's '.predict()' function.
            sequence_length (int): an integer value of seconds to generate.    

        Return:
            A numpy array of float32 audio data.
    '''
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

    return np.array(generated_audio,np.float32)



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
    seconds_to_generate = 5
    sequence_length = seconds_to_generate * default_sample_rate  # Adjust as needed for the desired length of audio    
    generated_audio = generate_audio(generator, seed_input, sequence_length)

    # Generate.
    open(output_filename, 'w').write(generated_audio)

    return output_filename



def save_base64_str_to_wav(b64str: str, output_filename: str) -> None:
    '''
        Helper function to convert a base64 encoded string to a .wav file and save it to disk.

        Args:
            b64str (str): a base64-encoded string that contains audio data.
            output_filename (str): the filepath where the .wav file will be written.

        Return:
            None.
    '''
    data = base64.b64decode(b64str)
    with wave.open(output_filename, "wb") as out_f:
        out_f.setnchannels(1)
        out_f.setsampwidth(2) # number of bytes
        out_f.setframerate(default_sample_rate)
        out_f.writeframesraw(data)
        out_f.close()




def validate_dataset(dataset_folder: str) -> None:
    '''
        Iterate over dataset .wav files and ensure none of them are non-unique (if so, delete them).
        This is necessary because Vital presets copy their wavetable data internally, so often the
        same audio data appears in multiple places. We'd like to train our model only on unique 
        data, though.

        Args:
            dataset_folder (str): folder containing the .wav files in the dataset used to train a LSTM model.

        Return:
            None.
    '''

    for root1, _, files1 in os.walk(dataset_folder):
        for filename1 in files1:
            for root2, _, files2 in os.walk(dataset_folder):
                for filename2 in files2:

                    try:
                        pt1 = os.path.join(root1,filename1)
                        pt2 = os.path.join(root2, filename2)

                        # If the first file is deleted, stop the inner iterator.
                        if (not os.path.exists(pt1)):
                            break

                        # Delete one file if both have different names but identical content.
                        if (not os.path.exists(pt2)
                            or pt1 == pt2 or os.path.getsize(pt1) != os.path.getsize(pt2) 
                            or not filecmp.cmp(pt1,pt2)):
                            continue
                        else:
                            os.remove(pt2)

                    except Exception as e:
                        print(str(e))
                        continue




def fill_dataset(vital_root_doc_folder: str) -> None:
    '''
        Recurse vital root presets folder and parse audio data in .vital preset files 
        into .wav files located in VyTools's '\dataset' folder.

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
    





# ==============
#     MAIN                        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('''
                Error! Provide string input arg for operation, either "DataAcquisition", "Training", or "Generation" and other necesssary args.
              ''')
        exit(-1)   

    start = time.perf_counter()

    # Run in one of 3 different modes:
    if sys.argv[1] == "DataAcquisition":
        if len(sys.argv) == 2:
            print('''
                    Error! If running the data acquisition code, also provide an arg for Vital's root preset directory (usually in the user's
                    "Documents" folder).
                ''')
            exit(-2)            
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder) 
        fill_dataset(sys.argv[2])
        validate_dataset(dataset_folder)
    # ===================================


    elif sys.argv[1] == "Training":
        reuse_existing_files: bool = False
        if len(sys.argv) == 2:
            print('''
                    NOTE: if running the training code, you may also provide a boolean arg for whether or not to reuse existing feature .bin files.
                ''')
        else:
            reuse_existing_files = sys.argv[2].lower().replace(' ', '') == 'true'
        if not reuse_existing_files:
            extract_features(dataset_folder, training_file, details_file)
        generate_model(training_file, model_file)
    # ===================================



    elif sys.argv[1] == "Generation":
        if len(sys.argv) == 2:
            print('''
                    Error! If running the .wav generation code, also provide an arg for the output file path.
                ''')
            exit(-3)         
        generate_wav(model_file, sys.argv[2])


    print('process finished. duration: ' + str(time.perf_counter() - start))