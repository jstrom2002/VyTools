## main.py -- written 3/18/2024 by JHS
##
## Command line tool to do all 3 stages
## of audio generation: building a dataset,
## model tranining, and running a generator
## on the trained dataset to produce new
## audio.
## ########################################

import os
import sys
import time
import VyToolsConsts
import VyToolsDataset
import VyToolsFeatures
import VyToolsModel

def working_dir(path: str) -> str:
    return os.path.join(os.path.dirname(__file__),path)

def print_red(text: str) -> None:
    print('\x1b[1;37;41m' + text + '\x1b[0m')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_red('''
                --- ERROR! Invalid args. ----
                Input arg format: `python.exe main.py <mode string> <output filepath> <other args>`
                Mode can be either "DataAcquisition", "Training", or "Generation".
                If "Training", also include a boolean <use existing file> as an extra argument.
                If "Generation", also include a filepath to the model file you wish to use as an argument.
              ''')
        exit(-1)   

    start = time.perf_counter()

    # Run in one of 3 different modes:
    if sys.argv[1] == "DataAcquisition":
        if len(sys.argv) == 2:
            print_red('''
                    Error! If running the data acquisition code, also provide an arg for Vital's root preset directory (usually in the user's
                    "Documents" folder).
                ''')
            exit(-2)            
        print('Parsing Vital preset files and saving audio dataset to: ' + sys.argv[2])         
        if not os.path.exists(working_dir(VyToolsConsts.DATA_PATH)):
            os.makedirs(working_dir(VyToolsConsts.DATA_PATH)) 
        VyToolsDataset.fill_dataset(sys.argv[2])
        VyToolsDataset.validate_dataset(working_dir(VyToolsConsts.DATA_PATH))
    # ===================================



    elif sys.argv[1] == "Training":
        reuse_existing_files = sys.argv[3].lower().replace(' ', '') == 'true'
        if not reuse_existing_files:
            VyToolsFeatures.extract_features(
                working_dir(VyToolsConsts.DATA_PATH), 
                working_dir(VyToolsConsts.TRAINING_PATH), 
                working_dir(VyToolsConsts.DETAILS_PATH)
            )
        if not os.path.exists(working_dir(VyToolsConsts.TRAINING_PATH)):
            print_red(' --- ERROR! No extracted feature .bin file found. ---- ')
            exit(-3)   
        else:
            print('Generating trained AI model and saving to: ' + sys.argv[2])         
            VyToolsModel.generate_model(
                sys.argv[2], 
                working_dir(VyToolsConsts.DATA_PATH), 
                working_dir(VyToolsConsts.TRAINING_PATH),
                VyToolsConsts.LATENT_SIZE
            )
    # ===================================



    elif sys.argv[1] == "Generation":
        if len(sys.argv) < 4:
            print_red('''
                    Error! If running the .wav generation code, also provide an arg for the output file path and a path to the model file
                    you wish to use.
                ''')
            exit(-4)
        print('Loading model and running code to generate wav file to: ' + sys.argv[2])         
        VyToolsModel.generate_wav(sys.argv[2], sys.argv[3], VyToolsConsts.LATENT_SIZE, VyToolsConsts.SAMPLE_RATE)
    # ===================================



    print('process finished. duration: ' + str(time.perf_counter() - start))