# Concentration Detection using EEG Signals
This is a simple brain-computer interface project to detect user concentration.

The project is divided into three crucial steps.

## 1. Data Collection

In collecting EEG data, it is crucial to design experiments to record data by taking into account considerations such as eye blinks and mental fatigue. As blinking impacts the EEG signal detected, it is important to either prevent the presence of blinks or training a model to be robust to blinks. Since this project was completed over the course of a few days, it was easier to prevent the presence of blinks by instructing the subject not to blink, but also to allow for rest periods during data collection where the subject could blink. Thus, the experiment designed is as follows:

    1. A sound signifying the start of the experiment plays.
    2. A 2-second rest period starts and finishes.
    3. A 2-second recording period begins where all EEG signals are recorded, and the period finishes.
        - Here, the subject is instructed to perform the required task, either concentrating or doing nothing.
    4. A sound is played immediately after to signify the end of a single collection loop.
    5. Loop back to step 2 until the specified number of loops finishes.
    6. A sound is played to signify the end of the experiment.

Notably, since the subject would be instructed not to blink during step 3, a down-time in step 2 allows the subject to blink and do whatever they need to prevent unnecessary movements during the recording process. The sound being played in step 4 allows the subject to stop focusing on the task at hand and prevent mental strain.

A python notebook file called `data-collection.ipynb` is available in the repository. In my project, I used this data-collection loop to collect 150 data points of doing nothing and 150 data points of the subject concentrating. This data is saved to a file to be used in step 2.

After this step, we will have two arrays, one being the recorded brain waves and the other being the associated actions. They will have shape `[B, C, T]` and `[B,]`, respectively.

Here, B refers to the number of samples we have. C refers to the electrodes of the recording device, and T refers to the timesteps. If the sampling frequency is 256 Hz and you record for 2 seconds, T will be 512 (256 * 2), because the device captures ~256 data points per second.

## 2. Data Processing

After data is collected, they should now be processed for use in modeling. 

It should be noted that, with traditional machine learning models, it is usual to manually extract features of interest. With neural networks, however, they are able to automatically extract relevant features to best classify the brain waves. In this repository, traditional machine learning models are used, hence, this preprocessing step is required.

The data processor first creates a frequency-domain representation of the signal by estimating the power spectral density using Welch's method. In basic terms, it tells us what frequencies are present in the brain waves, and also the strength of those signals. This allows us to then extract the strength of frequencies of interest, which may correspond to different actions the brain is performing. 

The processor then extracts the band power of the brain's alpha, beta, gamma, theta, and delta waves. Note that since the Muse 2 (which was originally used for this experiment) has 5 electrodes (but 1 is irrelevant), we have to extract those waves for each of the channel. Since we are using 4 electrodes only, there will be 4 * 5 (5 types of waves) features available (I'll link to my other GitHub repository in the future to discuss this!).

After this step, the features (extracted bandpower) will now have shape `[B, C, 5]`.


## 3. Data Modeling

In this step, the data modeler is used to simply reshape the data (it isn't necessary) from the previous step. We then convert these data into Pandas dataframes and fit a scikit-learn Gradient Boosting classifier on it. Here, cross-validation is used to determine model's generalizability, giving us accuracies greater than 80%. Note, since we only trained using one subject's brainwaves, it is unlikely that this model will perform well with other people's brainwaves.

The model is saved and then can be used for predictions. To use it in a real-time setting, a loop similar to the data-collection process is recommended, where 2 seconds are recorded, then passed through a model for detection, and feedback is performed.

## Additional Notes
In this project, a Muse 2 was used to record brainwave signals. An lab streaming layer to stream data from Muse 2 to the device was created using `BlueMuse` (https://github.com/kowalej/BlueMuse), and the lab streaming layer was integrated into Python using the library `pylsl`.




