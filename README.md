# Concentration Detection using EEG Signals
This is a simple brain-computer interface project to detect user concentration.

The project is divided into three crucial steps.

1. Data Collection

In collecting EEG data, it is crucial to design experiments to record data by taking into account considerations such as eye blinks and mental fatigue. As blinking impacts the EEG signal detected, it is important to either prevent the presence of blinks or training a model to be robust to blinks. Since this project was completed over the course of a few days, it was easier to prevent the presence of blinks by instructing the subject not to blink, but also to allow for rest periods during data collection where the subject could blink. Thus, the experiment designed is as follows:

    1. A sound signifying the start of the experiment plays.
    2. A 2-second rest period starts and finishes.
    3. A 2-second recording period begins where all EEG signals are recorded, and the period finishes.
        - Here, the subject is instructed to perform the required task, either concentrating or doing nothing.
    4. A sound is played immediately after to signify the end of a single collection loop.
    5. Loop back to step 2 until the specified number of loops finishes.
    6. A sound is played to signify the end of the experiment.

Notably, since the subject would be instructed not to blink during step 3, a down-time in step 2 allows the subject to blink and do whatever they need to prevent unnecessary movements during the recording process. The sound being played in step 4 allows the subject to stop focusing on the task at hand and prevent mental strain.

In my project, I used this data-collection loop to collect 150 data points of doing nothing and 150 data points of the subject concentrating. This data is saved to a file to be used in step 2.

2. Data Processing
3. Data Modeling





