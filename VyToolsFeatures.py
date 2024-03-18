import os
import VyToolsConsts
import VyToolsShared


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

    # Reset training files by default.
    features = open(feature_filename,'wb')
        
    # Open files and stream float32/int32 data into them continuously.
    file_count = 0
    float32_count = 0
    for root, _, files in os.walk(dataset_folder):
        for filename in files:    
            if filename.endswith('.wav'):
                abs_filepath = os.path.join(root,filename)
                ifile = VyToolsShared.load_audio(abs_filepath) 
                if 'data' in ifile.keys():
                    file_count = file_count + 1
                    float32_count = float32_count + len(ifile['data'])
                    for x in ifile['data']:
                        features.write(x)
    features.close()


    # Finally, include a details.txt file with some useful dataset details.
    details = open(details_filename,'w')
    details.write(str({
        'dataset_sample_size': VyToolsConsts.LATENT_SIZE,
        'feature_filename': feature_filename,
        'float32_count': float32_count,
        'file_count': file_count,
    }))
    details.close()

