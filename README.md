# Speech and Text Processing

Deep learning has revolutionized speech and text processing, taking it to new heights. But how do we build speech recognition models? It's not as straightforward as it seems. There are different approaches, each with its own complexities. Let's dive into building three different models for three different types of speech inputs.
1. Frame Level Speech Recognition
2. Automatic Speech Recognition: Utterance to Phoneme transcription 
3. Attention-based end-to-end speech-to-text model

## 1. Frame Level Speech Recognition

Speech data consists of audio recordings, while phonemes represent the smallest sound units ("OH", "AH", etc.). Spectrograms, particularly MelSpectrograms, are commonly used to visually represent speech signals' frequency changes over time. In our dataset, we have audio recordings (utterances) and their corresponding phoneme state (subphoneme) labels from the Wall Street Journal (WSJ).

**Inputs:** Raw Mel Spectrogram Frame  
**Outputs:** Frame Level Phoneme State Labels

**Phonemes Example:** ["+BREATH+", "+COUGH+", "+NOISE+", "+SMACK+", "+UH+", "+UM+", "AA", "AE"], and so on, with 40 phoneme labels.
</br>
<img width="383" alt="image" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/559af2a0-7e3d-4447-8683-1091779dc8d2">
</br>
Phonemes are like the building blocks of speech data. One powerful technique in speech recognition is modeling speech as a Markov process with unobserved states, known as phoneme states or subphonemes. Hidden Markov Models (HMMs) estimate parameters to maximize the likelihood of observed speech data.

Instead of HMMs, we're taking a model-free approach using a Multi-Layer Perceptron (MLP) network to classify Mel Spectrograms and output class probabilities for all 40 phonemes.

**Feature Extraction**

Each utterance is converted into a Mel Spectrogram matrix of shape (*, 27) after performing Short-Time Fourier Transform (STFT) on small, overlapping segments of the waveform.</br>
<img width="276" alt="feature_extraction" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/9184dabc-6031-47b9-97ce-c8097a6c1c22">
<img width="367" alt="feature_extraction2" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/cc9a9910-2650-4402-aa57-86bf5714ce41">

**Context**

To ensure accurate predictions, we provide context around each vector. For example, a context of 5 means appending 5 vectors on both sides, resulting in a vector of (11, 27).

<img width="248" alt="context" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/e70ec5a5-4a9e-47a9-b1da-301432b329e0">

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


# 2. **Automatic Speech Recognition: Utterance to Phoneme transcription**

In this problem, we'll utilize a neural network to process an audio recording where a person says the word "yes," resulting in the phonetic transcription /Y/ /EH/ /S/. Our focus will be on implementing RNNs along with the dynamic programming algorithm known as Connectionist Temporal Classification to produce these labels.

## Problem
Standard speech recognition often involves labeling each frame (time step) of the recording with a phoneme. However, spoken words have variable lengths, making this approach unnatural. We want to directly output the phoneme sequence without worrying about exact timing.

<img width="291" alt="image" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/dd80ca1d-620b-4361-9f6f-34ce3064ed23">


## Challenge:
Converting a variable-length speech recording (represented as feature vectors) into a sequence of phonemes with different lengths and no one-to-one correspondence in timing. without direct temporal correspondence can be referred to as generating order-aligned time-asynchrony labels. PyTorch provides functions like pad_sequence(), pad_packed_sequence(), and pack_padded_sequence() for padding and packing variable-length sequences efficiently.

<img width="262" alt="image" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/b7bae84e-b62c-43df-9e16-133c24c5c1ec">

## A two-stage approach:

1. **RNN network predicts probabilities for each phoneme at every time step.**
2. **CTC algorithm performs dynamic programming to generate the final phoneme sequence from the probabilities.**

<img width="385" alt="image" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/605c39a7-7fe2-4cad-8cdc-5ffae2462e75">

### RNNs

Their ability to capture temporal dependencies makes them suitable for analyzing sequential data like speech.

### CTC

- Decoding probabilities from the RNN's output at every time step.
- Employing dynamic programming to find the most likely phoneme sequence based on the probabilities.
- Utilizing a "blank" symbol to handle silent regions and repetitions.

  <img width="374" alt="image" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/f31730fa-47fb-4e30-ab90-4a1348343199">


## Building the Model

The model is an RNN that processes the speech feature vectors and outputs a sequence of probability vectors for each phoneme (including a blank symbol) at each time step. The architecture involves:

- 1D convolutional layers (CNNs) to capture local dependencies in the speech features.
- Bidirectional LSTMs (BLSTMs) to capture long-term contextual information.
- Pyramidal LSTMs (pBLSTMs) for potential downsampling of the input sequence.
- A final layer with softmax activation converts the hidden representations into phoneme probabilities.

<img width="373" alt="image" src="https://github.com/Santhoshkumar-p/speech-processing/assets/24734488/60033f00-a41a-42c6-afc8-52b930728dec">

## Training the Network

Training data consists of speech recordings and their corresponding phoneme sequences. Since the target phoneme sequence is shorter and asynchronous compared to the input, we need a way to compute the loss function for training.

- **Viterbi Training:** Finds the single most likely alignment between the phoneme sequence and the input using the Viterbi algorithm.
- **CTC Loss:** Calculates the expected loss over all possible alignments using dynamic programming (forward-backward algorithm). PyTorch's CTCLoss function can be used for this purpose.

Using this network along with appropriate speech data transformations such as Time Masking and Frequency Masking, we've managed to attain a Levenshtein Distance of 6.
This approach forms the foundation for various speech processing applications like voice assistants and automatic captioning.

## References

- [Course Website](https://deeplearning.cs.cmu.edu/S24/index.html): Deep Learning Course - CMU

By understanding these different approaches, we can harness the power of deep learning for speech recognition.
