
### Data Source and Methods

#### Data Source

The dataset utilized in this study was sourced from Kaggle, which is recognized as the world’s largest repository for datasets. Specifically, this dataset comprises EEG (electroencephalography) data derived from a study focused on investigating the corollary discharge process in individuals with schizophrenia and healthy controls.

#### Dataset Description

The dataset includes EEG recordings from 81 human subjects, divided into two groups: 32 healthy controls and 49 patients diagnosed with schizophrenia. The subjects engaged in a simple button-pressing task with three distinct conditions:
1. Pressing a button to immediately generate a tone.
2. Passively listening to the same tone.
3. Pressing a button without generating a tone.

The primary objective of the study was to examine the corollary discharge process, which pertains to the brain's capability to suppress responses to sensory consequences that result from one’s own actions.

#### Key Findings

In the initial findings, it was observed that healthy controls exhibited suppression of the N100 ERP (event-related potential) component when they pressed a button to generate a tone, as compared to when they passively listened to the tone. Conversely, patients with schizophrenia did not display this suppression effect.

The current dataset represents a larger sample replication of a previous study, incorporating data from 22 controls and 36 patients, alongside an additional 10 controls and 13 patients from the earlier study.

#### Data Preprocessing

The EEG data underwent extensive preprocessing to ensure the quality and reliability of the findings. The preprocessing steps included:
- **Re-referencing**: Adjusting the reference point for EEG signals.
- **Filtering**: Applying filters to remove noise and irrelevant frequencies.
- **Artifact Removal**: Identifying and eliminating artifacts that could distort the EEG signals.
- **Averaging**: Deriving event-related potentials (ERPs) by averaging the EEG signals across multiple trials.

The preprocessed EEG data were then analyzed at 9 electrode sites to extract the relevant ERPs.

#### References

1. [Kaggle Dataset](https://www.kaggle.com/datasets/broach/button-tone-sz)

### Methods

#### Basic CNN Model

**Data Preprocessing**

Preprocessing was done in the following steps:
1. The demographic data, time samples, and electrode labels were read into Pandas dataframes and a dictionary was created to map each subject to their diagnostic group.
2. EEG data for each trial was averaged in chunks of 16 rows to reduce noise and computational complexity.
3. The reshaped data was converted into feature vectors suitable for CNN input and EEG data was normalized using maximum normalization.
4. Finally, the dataset was split into training and testing sets with an 80-20 ratio. The data was reshaped into 2D arrays required for CNN input.

**Model Training and Evaluation**

Following the preprocessing, a CNN model was trained and evaluated for schizophrenia detection. The CNN architecture featured:
- **Convolutional Layers**: Two convolutional layers with tanh activation, each followed by max-pooling layers for downsampling.
- **Dropout Layers**: Introduced to prevent overfitting, with dropout rates of 0.2.
- **Fully Connected Layers**: A fully connected layer with ReLU activation
- **Output Layer**: A single neuron with sigmoid activation for binary classification.

The model was trained using the following parameters:
- **Loss Function**: Binary cross-entropy.
- **Optimizer**: Adam with a learning rate of 0.0000075.
- **Batch Size**: 256.
- **Epochs**: 10.
- **Validation**: 20% of the data was used for validation during training.

The model achieved 59.83% training accuracy following results over 10 epochs.

#### Data Augmentation and Sliding Window Averaging

**Data Preprocessing and Feature Extraction**

The preprocessing and feature extraction of EEG data commenced with the acquisition of demographic information and EEG recordings from participants. The EEG data, stored in CSV files, contained event-related potential (ERP) data and trial data. To ensure data quality, noisy or incomplete trials were identified and removed prior to feature extraction.

Feature extraction involved several steps:
1. The ERP data were loaded into memory.
2. Trial data were extracted.
3. A sliding window averaging technique was applied to calculate the mean values of EEG signals over fixed intervals. This yielded a feature matrix suitable for training machine learning models.

**Data Augmentation**

To enhance model robustness and mitigate overfitting, data augmentation techniques were employed. Utilizing the ImageDataGenerator module from the Keras library, various augmentation transformations were applied to the EEG signals. These included rotation, shifting, shearing, zooming, and flipping, effectively increasing the training dataset's diversity.

**Model Training and Evaluation**

Before model training, the feature matrix underwent normalization using the normalize function from the scikit-learn library, ensuring that the features were scaled uniformly. A Convolutional Neural Network (CNN) was then constructed using TensorFlow's Keras API. The architecture comprised multiple convolutional layers followed by max-pooling layers for spatial downsampling. Dropout layers were incorporated to prevent overfitting, and fully connected layers facilitated feature aggregation and classification.

The training process optimized the binary cross-entropy loss function using the Adam optimizer. Early stopping and model checkpoint callbacks monitored validation accuracy, saving the best-performing model based on the validation set's performance. The trained CNN model was evaluated on an independent test dataset, achieving a classification accuracy of 68%.

#### Integration of Bandpass Filtering, Fast Fourier Transform, and Data Augmentation

**Data Preprocessing and Feature Extraction**

Rigorous preprocessing techniques were employed to prepare the EEG data. Initially, demographic data, ERPs, and trial-specific data were loaded into the computational environment. Several preprocessing functions ensured consistency and accuracy:
- **Interval Means Calculation**: The `calculate_means` function computed mean values of EEG signals over fixed intervals, facilitating data aggregation into manageable segments.
- **Averaging by Fixed Intervals**: The `averaged_by_N_rows` method computed mean values for every N rows in the EEG data, enabling granular analysis.
- **Bandpass Filtering**: A bandpass filter, implemented via the `butter_bandpass_filter` function, was applied with a low cut-off frequency of 0.5 Hz and a high cut-off frequency of 50 Hz. This step isolated relevant signal characteristics and reduced noise.
- **Frequency Domain Feature Extraction**: The Fast Fourier Transform (FFT) algorithm, implemented through the `extract_frequency_features` function, captured the spectral content of the EEG data.

**Data Augmentation**

EEG signals were treated as "pseudo-images" for augmentation. Using the ImageDataGenerator, transformations such as rotation, translation, scaling, shearing, and flipping were applied to these "pseudo-images." This approach augmented the dataset with synthetic samples, enhancing robustness and diversity.

**Model Training and Evaluation**

Following preprocessing and augmentation, a CNN model was trained and evaluated for schizophrenia detection. The CNN architecture featured:
- **Convolutional Layers**: Two convolutional layers with ReLU activation and batch normalization, each followed by max-pooling layers for downsampling.
- **Dropout Layers**: Introduced to prevent overfitting, with dropout rates of 0.3 and 0.4.
- **Fully Connected Layers**: Two fully connected layers with ReLU activation, followed by dropout layers to maintain regularization.
- **Output Layer**: A single neuron with sigmoid activation for binary classification.

The training process included callbacks for model checkpointing, early stopping, and learning rate reduction, ensuring convergence towards an optimal solution while preventing overfitting. The CNN model achieved a validation accuracy of 99%, demonstrating high efficacy in distinguishing between schizophrenia and non-schizophrenia cases. Performance metrics, including test loss and accuracy, validated the model's effectiveness. Additionally, the temporal evolution of training and validation accuracy over epochs was analyzed to assess convergence behavior and potential overfitting tendencies.
