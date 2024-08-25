# Backend Example E1

import numpy as np
from brainflow.data_filter import DataFilter, WindowOperations
import pickle
from pylsl import StreamInlet, resolve_stream, resolve_byprop, StreamInfo
import datetime as dt
from typing import Tuple, List
import time
import os
import serial
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
from pygame import mixer

mixer.init()

class DataRecorder():
    def __init__(self, sampling_frequency: int=256, sound_required: bool=True) -> None:
        self.sampling_frequency = sampling_frequency

        if sound_required:
            self.sound_dictionary = {
                "enter": mixer.Sound("sounds/enter.mp3"),
                "bell": mixer.Sound("sounds/bell.mp3")
            }
        else:
            self.sound_dictionary = {}
        self.samples = []
        self.actions = []

    def record_loop(self, duration: float=2.0, iterations: int=10, action: str = 'Nothing') -> None:
        '''
        Records a (duration) seconds EEG sample for (iterations) iterations.

        Outputs: None. But, appends to self.samples 10 (iterations) np.ndarrays of shape [5, duration*self.sampling_frequency]. 
                    5 corresponds to the number of electrode channels present in the Muse 2. (For usage, we have to remove the last channel, as it's irrelevant.)
        
        Call self.save_samples(filepath) to save samples and actions into one .npz file.
        '''
        # Test whether a stream can be found.
        stream = self._initiate_stream()
        if stream is None:
            print("No EEG stream was found.")
            return None
        
        # >> A stream has been found and connected.
        # This sound signifies the start of the trial.
        self._play_sound("bell")

        for i in range(iterations):
            time.sleep(2) # This 2 seconds duration is used for rest time, where no brain activity is recorded. This allows for participants to perform other actions, such as blinking and heavy breathing.
            sample = self._record_eeg_sample(stream, duration)
            sample = self._padtrim_sample(sample, self.sampling_frequency, duration)

            self.samples.append(sample) # Directly modifies instance list.
            self.actions.append(action) # Directly modifies instance list.

            self._play_sound("enter") # This indicates the end of a single recording.
        self._play_sound("bell") # This indicates the end of the trial.

    def collapse_sample(self) -> None:
        '''
        Converts samples and actions into numpy arrays without having to go through save_sample.
        '''
        self.samples = np.array(self.samples) # Expected Shape: (ITERATIONS, CHANNELS, TIMESTEPS)
        self.actions = np.array(self.actions) # Expected Shape: (ITERATIONS,)


    def save_sample(self, save_directory: str, single_action: bool=True) -> None: 
        '''
        Inputs: save_filepath (the directory of the to-be saved npz file).
                single_action (is a single action being recorded? If so, we can name the file according to the action.)

        Outputs: None. But, self.samples and self.actions are reset.
        '''
        print(save_directory)
        length = len(self.samples)
        first_action = self.actions[0]

        samples = np.array(self.samples) # Expected Shape: (ITERATIONS, CHANNELS, TIMESTEPS)
        actions = np.array(self.actions) # Expected Shape: (ITERATIONS,)

        current_time = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_filepath = f'{save_directory}/{length}_'
        print(save_filepath)
        # If a single action is being recorded, we can identify the file by that name too.
        if single_action:
            save_filepath += f"{first_action}_"
        
        save_filepath += f"{current_time}.npz"

        # Save into npz file with samples and actions.
        print(save_filepath)
        np.savez(save_filepath, samples=samples, actions=actions)

        self.samples = []
        self.actions = []

    def reset_sample(self) -> None:
        '''
        Resets the self.samples and self.actions lists.
        '''

        self.samples = []
        self.actions = []

    def _initiate_stream(self, timeout: int=4) -> StreamInfo | None:
        '''
        Initiates a stream with (timeout) timeout. If a stream has not been found within
        the specified timeout seconds, then it is assumed no stream is present.
        '''

        streams = resolve_byprop('type', 'EEG', timeout=timeout)
        if not streams:
            return None
        return streams
        
    def _record_eeg_sample(self, stream: StreamInfo, duration: float) -> np.ndarray:
        '''
        Inputs: stream channel and duration of recording (in seconds).
            Records a single EEG recording from (stream) stream channel for (duration) seconds.

        Outputs: A numpy array of shape [5, duration*sampling_frequency]. Where 5 refers to the recording channels in a Muse2.
        '''

        # Gets stream inlet.
        inlet = StreamInlet(stream[0])

        # Initialize required variables to calculate time elapsed and compile samples.
        start_time = dt.datetime.now()
        samples = []

        # Collect data for the given duration.
        while dt.datetime.now() - start_time < dt.timedelta(seconds=duration):
            sample, _ = inlet.pull_sample()
            samples.append(sample) 
        
        samples = np.array(samples) # Shape: [duration*sampling_frequency, 5] (5 channels from Muse 2). Example shape: [512, 5]
        samples = samples.T # Shape: [5, duration*sampling_frequency]. Example shape: [5, 512]
        samples = np.ascontiguousarray(samples) # Converts array to contiguous array for faster processing.
        return samples
    
    def _play_sound(self, sound_name: str):
        if sound_name in self.sound_dictionary:
            self.sound_dictionary[sound_name].play()

    def _padtrim_sample(self, samples: np.ndarray, sampling_frequency: int, duration: float) -> np.ndarray:
        # Get the number of channels and timesteps.
        C, T = samples.shape

        # Calculate the desired length.
        target_length = int(sampling_frequency * duration)

        # If the sample is shorter than the desired length, pad it.
        if T < target_length:
            pad_width = ((0, 0), (0, target_length - T))
            samples = np.pad(samples, pad_width, 'constant')
        else:
            # Else, if the sample is longer than the desired length, trim it.
            samples = samples[:, :target_length]
        
        # Return the padded/trimmed sample.
        return samples

class DataProcessor():
    def __init__(self, directory: str='data', sampling_frequency: int=256) -> None:
        # Loads the directory to parse .npz files from.
        self.directory = directory
        self.sampling_frequency = sampling_frequency

        self.samples = []
        self.actions = []

        self.bands  = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 100)
        }

    def extract_bandpowers(self, collapse=False) -> np.ndarray:
        '''
        Extracts band powers from loaded samples.

        Inputs: None, but utilizes self.samples and self.sampling_frequency.
            - self.samples (np.ndarray): The input sample(s). It can be a 3D array for batched samples or a 2D array for an individual sample.
                Shape: [B, C, T] or [C, T]. B -> Batch size, C -> EEG channels, T -> Timesteps.
            - self.sampling_frequency (int): The sampling frequency of the samples. Default is 256 for the Muse 2. 
        
        Output: None, but modifies self.extracted_bp_samples.
            -self.extracted_bp_samples <- np.ndarray: An array containing the extracted band powers for each sample and channel.
                Shape: [B, C, 5]. 5 -> extracted band_power for each brainwave type (delta, theta, alpha, beta, gamma).   
        '''
        # Get the dimensions of the samples array
        B, C, T = self.samples.shape  

        # Remove the last channel if the number of channels is 5. (Muse2 usually gives 5 channel recordings, but the last channel is irrelevant) 
        # This would likely have already been caught by the _load_data method.
        if C == 5:
            self.samples = self.samples[:, :4, :]
            C = 4

        # Initialize an empty list to store the extracted features.
        # Iterate over each sample and each channel in the sample and calculate the bandpowers for each brainwave type.
        extracted_features = []  
        for b in range(B):  
            sample = self.samples[b]  
            extracted_channels = []  

            for c in range(C):  
                channel = sample[c]  
                psd = DataFilter.get_psd_welch(channel, nfft=self.sampling_frequency, overlap=self.sampling_frequency//2, sampling_rate=self.sampling_frequency, window=WindowOperations.HANNING.value)  # Compute the power spectral density using Welch's method
                extracted_band_powers = []  

                for band_name, (low, high) in self.bands.items(): 
                    band = DataFilter.get_band_power(psd, low, high)
                    extracted_band_powers.append(band) 

                extracted_channels.append(extracted_band_powers)  

            # Extracted_channels would be of shape [5,]
            extracted_features.append(extracted_channels) 

        # Extracted_features would be of shape [B, C, 5]
        extracted_features = np.array(extracted_features)

        if collapse:
            extracted_features = extracted_features.reshape((B, -1))
        return np.array(extracted_features)  

    def get_actions(self) -> np.ndarray:
        return self.actions

    def load_data_direct(self, samples: np.ndarray, actions: np.ndarray, delete_last_eeg_channel: bool=True) -> None:
        self.samples = np.array(samples)

        # Here, we would like to remove the last recording channel, as it is irrelevant for Muse2.
        if delete_last_eeg_channel:
            self.samples = np.delete(self.samples, -1, axis=1)

        self.actions = np.array(actions)

    def load_data(self, delete_last_eeg_channel: bool=True) -> None:
        '''
        Loads all the .npz data from the initialized directory. Removes the last EEG channel (in dim 1) if delete_last_eeg_channel is True.

        Outputs: None, but modifies self.samples and self.actions to include all the files.
        '''
        npz_files = [file for file in os.listdir(self.directory) if file.endswith(".npz")]

        samples = []
        actions = []

        for npz_file in npz_files:
            data = np.load(os.path.join(self.directory, npz_file))
            samples.extend(data["samples"])
            actions.extend(data["actions"])

        self.samples = np.array(samples)

        # Here, we would like to remove the last recording channel, as it is irrelevant for Muse2.
        if delete_last_eeg_channel:
            self.samples = np.delete(self.samples, -1, axis=1)

        self.actions = np.array(actions)
     
class DataModeler():
    def __init__(self, directory: str) -> None:
        self.directory = directory

        self.samples = []
        self.actions = []
        self.model = None

    def load_data_direct(self, samples: np.ndarray, actions: np.ndarray)-> None:
        '''
        Allows user to load data directly from numpy arrays rather than files.
        '''
        self.samples = np.array(samples)
        self.actions = np.array(actions)

    def load_data(self, filename: List[str], verbose: bool=True) -> None:
        '''
        Loads data from a list of filenames. Aggregates all of them into self.samples and self.actions.
            Verbose indicates whether error messages will be printed.
        '''
        if type(self.samples) is list:
            for name in filename:
                path = os.path.join(self.directory, name)
                data = np.load(path)

                self.samples.append(data['extracted_features'])
                self.actions.append(data['actions'])

            self.samples = np.array(self.samples)
            self.actions = np.array(self.actions)
        else:
            if verbose:
                print("Error: The data has already been loaded. If you want to add multiple filenames, please provide them in a list. Reset the variables with .reset_samples and please try again.")
     
    def reset_samples(self) -> None:
        self.samples = []
        self.actions = []

    def reshape_samples(self, target_shape: Tuple[int,...], verbose: bool=True) -> None:
        sample_shape = self.samples.shape
        self.samples = np.reshape(self.samples, target_shape)


    def load_model(self, model_filepath: str) -> None:
        '''
        Loads a pre-trained model from the given file.

        Inputs:
        - filename (str): The name of the file containing the pre-trained model.
        
        Output:
        - Any: The pre-trained model.
        '''
        with open(model_filepath, 'rb') as f:  # Open the file in read mode
            self.model = pickle.load(f)  # Load the model from the file

    def predict(self, verbose: bool=True) -> np.ndarray:
        '''
        Predicts the mental state of the user based on the given EEG samples.

        Inputs: None, but utilizes self.samples
        - samples (np.ndarray): The input EEG samples.
            Shape: [C, T]. C -> EEG channels, T -> Timesteps.

        Output:
        - np.ndarray, which contains the predicted mental states of the user ('concentrating' or 'nothing').
        '''
        if self.model is None:
            if verbose: 
                print("No model has been loaded. Please load a model via load_model method first.")
                return
    
        res = self.model.predict(self.samples)  # Make predictions using the model

        return res  # Returns the prediction.
    

class ArduinoInterface():
    def __init__(self) -> None:
        try:
            self.arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)  # Use the correct COM port for Windows
        except:
            self.arduino = None

    def send_signal(self, state) -> None:
        '''
        Writes messages to Arduino if the port is connected and open.
        '''
        if self.arduino is not None and self.arduino.is_open():
            if state == 'concentrating':
                self.arduino.write(b'C')  # Send 'C' to Arduino
            elif state == 'nothing':
                self.arduino.write(b'N')  # Send 'N' to Arduino