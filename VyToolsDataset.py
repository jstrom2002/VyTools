## VyToolsDataset.py -- written by JHS 3/15/2024
##
## Utility functinos for populating a dataset
## from the installed Vital root directory.
## #############################################

import base64
import filecmp
import json
import os
import VyToolsConsts
import VyToolsShared

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
    extracted_count = len([name for name in os.listdir(VyToolsConsts.DATA_PATH) if os.path.isfile(name)]) + 1
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
                                            VyToolsShared.write_raw_to_wav(VyToolsConsts.DATA_PATH + 
                                                                                '\\extracted' + str(extracted_count) + '.wav', cmp['audio_file'])                                            
                                            extracted_count = extracted_count + 1
                                        if 'keyframes' in cmp.keys() and cmp['keyframes'] and len(cmp['keyframes']) > 0:
                                            for kf in cmp['keyframes']:
                                                if 'wave_data' in kf.keys():
                                                    VyToolsShared.write_raw_to_wav(VyToolsConsts.DATA_PATH + 
                                                                                        '\\extracted' + str(extracted_count) + '.wav', kf['wave_data'])
                                                    extracted_count = extracted_count + 1
                    except Exception as e:
                        print(str(e))                    
    


