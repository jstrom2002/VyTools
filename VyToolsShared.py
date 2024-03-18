import numpy as np
import wave
import VyToolsConsts


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

    # Normalise float32 array so that values are between -1.0 and +1.0 by dividing 
    # by the max int16 val.                                                      
    audio_normalized = audio_as_np_float32 / (np.iinfo(np.int16).max)   

    # Keep only the same small number of N samples of each wav sample. If an audio
    # file contains too few samples, repeat the original sample N times until it's
    # large enough.
    while len(audio_normalized) < VyToolsConsts.LATENT_SIZE:
       audio_normalized = np.concatenate([audio_normalized, audio_normalized])
    audio_normalized = audio_normalized[:VyToolsConsts.LATENT_SIZE]

    assert(len(audio_normalized) >= VyToolsConsts.LATENT_SIZE)

    return {
        'channels': ifile.getnchannels(),
        'samples': ifile.getnframes(),
        'data': audio_normalized
    }


def write_raw_to_wav(output_filename: str, data: bytes) -> None:
    '''
        Write raw int16 bytes to .wav file.

        Args:

        Return:
            None.
    '''
    with wave.open(output_filename, "wb") as out_f:
        out_f.setnchannels(1) # mono sound
        out_f.setsampwidth(2) # number of bytes == 2, ie 16-bit
        out_f.setframerate(VyToolsConsts.SAMPLE_RATE)
        out_f.writeframesraw(data)
        out_f.close()


def write_float_to_wav(output_filename: str, data: np.array, scale_factor: int = np.iinfo(np.int16).max) -> None:
    '''
        Convert to float32 to int16 and write to .wav file.

        Args:

        Return:
            None.
    '''
    with wave.open(output_filename, "wb") as out_f:
        out_f.setnchannels(1) # mono sound
        out_f.setsampwidth(2) # number of bytes == 2, ie 16-bit
        out_f.setframerate(VyToolsConsts.SAMPLE_RATE)
        out_f.writeframesraw((data * scale_factor).astype(np.int16))
        out_f.close()