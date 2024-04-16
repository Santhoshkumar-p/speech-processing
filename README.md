# Speech and Text Processing

Deep learning has revolutionized speech and text processing, taking it to new heights. But how do we build speech recognition models? It's not as straightforward as it seems. There are different approaches, each with its own complexities. Let's dive into building three different models for three different types of speech inputs.

## Frame Level Speech Recognition

Speech data consists of audio recordings, while phonemes represent the smallest sound units ("OH", "AH", etc.). Spectrograms, particularly MelSpectrograms, are commonly used to visually represent speech signals' frequency changes over time. In our dataset, we have audio recordings (utterances) and their corresponding phoneme state (subphoneme) labels from the Wall Street Journal (WSJ).

**Inputs:** Raw Mel Spectrogram Frame  
**Outputs:** Frame Level Phoneme State Labels

**Phonemes Example:** ["+BREATH+", "+COUGH+", "+NOISE+", "+SMACK+", "+UH+", "+UM+", "AA", "AE"], and so on, with 40 phoneme labels.
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dbb9c8c6-4af8-4a75-a655-6db75123217d/f3ad74b9-8657-4ac0-a948-72c23a432943/Untitled.svg)
Phonemes are like the building blocks of speech data. One powerful technique in speech recognition is modeling speech as a Markov process with unobserved states, known as phoneme states or subphonemes. Hidden Markov Models (HMMs) estimate parameters to maximize the likelihood of observed speech data.

Instead of HMMs, we're taking a model-free approach using a Multi-Layer Perceptron (MLP) network to classify Mel Spectrograms and output class probabilities for all 40 phonemes.

**Feature Extraction**

Each utterance is converted into a Mel Spectrogram matrix of shape (*, 27) after performing Short-Time Fourier Transform (STFT) on small, overlapping segments of the waveform.
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dbb9c8c6-4af8-4a75-a655-6db75123217d/497ac4a9-e7da-462d-ba8e-8778b08e823c/Untitled.png)
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dbb9c8c6-4af8-4a75-a655-6db75123217d/73d61b7d-c443-4c1b-9e98-2387633a3c29/Untitled.png)
**Context**

To ensure accurate predictions, we provide context around each vector. For example, a context of 5 means appending 5 vectors on both sides, resulting in a vector of (11, 27).

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/dbb9c8c6-4af8-4a75-a655-6db75123217d/bc9b0a7d-ebb9-4283-8ec5-532e0f5c3d6d/Untitled.png)

**Cepstral Normalization**

Cepstral Normalization helps remove channel effects in speaker recognition. It involves subtracting the mean and dividing by the standard deviation for each coefficient.

**Building the Network**

We used a Pyramid MLP architecture, achieving 88% classification accuracy. Various hyperparameters were considered in the process.

| Hyperparameters      | Values Considered                                            | Chosen            |
|----------------------|--------------------------------------------------------------|-------------------|
| Number of Layers     | 2-8                                                          | 8                 |
| Activations          | ReLU, LeakyReLU, softplus, tanh, sigmoid, Mish, GELU        | GELU              |
| Batch Size           | 64, 128, 256, 512, 1024, 2048                               | 1024              |
| Architecture         | Cylinder, Pyramid, Inverse-Pyramid, Diamond                  | Pyramid           |
| Dropout              | 0-0.5, Dropout in alternate layers                           | 0.25              |
| LR Scheduler         | Fixed, StepLR, ReduceLROnPlateau, Exponential, CosineAnnealing | CosineAnnealing |
| Weight Initialization | Gaussian, Xavier, Kaiming(Normal and Uniform), Random, Uniform | Kaiming          |
| Context              | 10-50                                                        | 20                |
| Batch-Norm           | Before or After Activation, Every layer or Alternate Layer or No Layer | Every Layer |
| Optimizer            | Vanilla SGD, Nesterov’s momentum, RMSProp, Adam               | AdamW             |
| Regularization       | Weight Decay                                                 | -                 |
| LR                   | 0.001                                                        | 0.001             |
| Normalization        | Cepstral Normalization                                       | Cepstral Normalization |



By understanding these different approaches, we can harness the power of deep learning for speech recognition.
