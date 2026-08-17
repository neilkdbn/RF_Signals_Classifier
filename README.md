
Intelligent RF Signal Classification Pipeline
 Overview
The Intelligent RF Signal Classification Pipeline is an advanced, 4-stage machine learning architecture designed for Automatic Modulation Classification (AMC) and robust spectrum sensing in highly dynamic, non-cooperative spectral environments.

Moving beyond traditional feature-based extractors, this system leverages deep learning to directly extract latent hierarchical features from raw Radio Frequency (RF) signals. It is explicitly engineered to handle severe channel fading, out-of-distribution signals, active adversarial spoofing, and the computational constraints of edge deployment.

 Architectural Framework
The pipeline is structured into four specialized, sequential stages:

Stage 1: Ingestion & Dataset Agent

Synthetic Generation: Uses MATLAB's Communications Toolbox to simulate realistic RF channels, deliberately applying Rician multipath fading, Doppler shifts, and Carrier Frequency Offsets (CFO).

OTA Capture: Bridges the simulation-to-reality gap by integrating over-the-air (OTA) hardware captures using SDR platforms like ADALM-PLUTO or USRP.

Stage 2: Dual-Representation Preprocessing

Time-Domain Branch: Processes and normalizes raw In-Phase (I) and Quadrature (Q) complex sequences to retain high-fidelity phase relationships.

Time-Frequency Branch: Computes Kaiser-windowed Short-Time Fourier Transforms (STFT) to generate 2D grayscale spectrograms, eliminating the blind spots of single-domain approaches.

Stage 3: Adaptive CNN-Transformer Core

Spatial Feature Extraction: Utilizes 1D and 2D convolutional layers to extract structural constellation features from the preprocessed data.

Temporal Modeling: Employs a Transformer encoder with sparse, local-windowed attention to model long-range temporal dependencies while strictly minimizing edge inference latency and parameter overhead.

Stage 4: Spectrum Cop Decision Engine

Outlier Detection: Deploys Minimum Covariance Determinant (MCD) and k-means clustering to dynamically flag unseen, unmodeled jammers or waveforms.

Continual Learning: Integrates Elastic Weight Consolidation (EWC) loss to continuously update network weights on new modulations without triggering catastrophic forgetting.

Explainable AI (XAI): Implements Grad-CAM visual highlighting over spectrograms to provide human operators with transparent verification of spectral classification decisions.


