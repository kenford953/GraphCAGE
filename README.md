# Graph Capsule Aggregation for Unaligned Multimodal Sequences
Code for Graph Capsule Aggregation (GraphCAGE), ICMI 2021, https://arxiv.org/pdf/2108.07543.pdf

This repository includes data, code and pretrained models for the ICMI 2021 paper, "Graph Capsule Aggregation for Unaligned Multimodal Sequences". In addition, we show details about cross-modal transformer, hyper-parameters and the extracted features at the end of this document.

## Data
Data files ("mosei_senti_data_noalign.pkl" and "mosi_data_noalign.pkl") can be downloaded from [here](https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0).

To retrieve the meta information and the raw data, please refer to the [SDK for these datasets](https://github.com/A2Zadeh/CMU-MultimodalSDK).

## Run the Code
### Requirements
- Python 3.7
- Pytorch 1.1.0
- numpy 1.19.2
- sklearn

### Train and test
~~~~
python main.py [--FLAGS]
~~~~

Note that the defualt arguments are for unaligned version of MOSEI. For other datasets, please refer to Supplmentary.

## Cross-modal Transformer
Cross-modal Transformer is proposed in [**Multimodal Transformer for Unaligned Multimodal Language Sequences**](https://arxiv.org/pdf/1906.00295.pdf)<br>, which enables one modality for receiving information from another modality. Thus, it can explicitly explore inter-modal dynamics.

## Extracted Features
The details about the extracted features of each modality are as follows:

Textual features are extracted from video transcripts by Glove word embeddings. Each sentence is converted to a sequence with length 50 which includes paddings and word embeddings. The dimension of the embedding is 300.

Acoustic features are extracted by COVAREP which indicates 12 Mel-frequency cepstral coefficients (MFCCs), pitch tracking, glottal source parameters, peak slope parameters and maxima dispersion quotients.

Visual features are extracted by Facet which indicate 35 facial action units. These units can record facial muscle movement for representing per-frame basic and advanced emotions.

## Hyperparameters
### CMU-MOSI
- Transformers Hidden Unit Size : 30
- Crossmodal Blocks : 5
- Crossmodal Attention Heads : 5
- Temporal Convolution Kernel Size : 1
- Embedding Dropout : 0.25
- Output Dropout : 0
- Batch Size : 16
- Initial Learning Rate : 1e-3
- Optimizer : RMSprop
- Epochs : 20
- Dimension of Capsules : 32
- Iteration of Routing : 2
- Weight of L2 Regularization : 1e-4

### CMU-MOSEI
- Transformers Hidden Unit Size : 30
- Crossmodal Blocks : 5
- Crossmodal Attention Heads : 5
- Temporal Convolution Kernel Size : 1
- Embedding Dropout : 0.25
- Output Dropout : 0.25
- Batch Size : 16
- Initial Learning Rate : 1e-3
- Optimizer : RMSprop
- Epochs : 20
- Dimension of Capsules : 64
- Iteration of Routing : 2
- Weight of L2 Regularization : 1e-4
