# End-to-End Project Report: 2D RF Signal Classification

This document is the complete narrative of the `feature/2d-models` branch, starting from the very first commit to the final merge. It explains exactly what we did, the struggles we faced, and the engineering rationale behind every major pivot.

---

## 1. Foundation & Architecture Initialization
**Commits:**
* `a156a0c` - Setup GitHub Actions CI pipeline and 2D tensor shape validation
* `fa96663` - Implement baseline ResNet-18 architecture for 2d spectrograms
* `547f286` - Build advanced STFT-RADN model integrating CBAM and residual dense blocks

**The Narrative & Rationale:**
We started the `feature/2d-models` branch by establishing a strict DevOps test suite (`test_2d_shapes.py`) to ensure our STFT spectrograms were formatting correctly. 

We then built two models:
1. **ResNet-18**: A standard, proven computer vision backbone. We used this as our "baseline" 2D model to see if standard image classification techniques would work on RF signals.
2. **STFT-RADN**: A highly advanced custom architecture using Convolutional Block Attention Modules (CBAM). The rationale here was that RF signals have localized "features" in the frequency domain, so an attention mechanism might outperform standard convolutions.

## 2. Infrastructure & Local Training Struggles
**Commits:**
* `c404bf3` - Implement 2d training loop with adamw and validation early stopping
* `26f9e3c` - Add latency/parameter benchmarking and ONNX export for Grad-CAM
* `0f527fc` - Optimize 2D training pipeline to mitigate overfitting
* `9d1f9bd` - Fix underfitting: remove over-regularization

**The Narrative & Rationale:**
We built the training loop and added ONNX exports so you could run MATLAB Grad-CAM validations later. However, we immediately hit a wall during local training:
* **The Speed Issue:** Training on a local CPU took over **300 seconds per epoch**. It was mathematically impossible to finish training and hyperparameter tuning before your submission deadline.
* **The Accuracy "Fluff":** When we did wait for epochs to finish, the model's behavior was erratic. We tried applying heavy Dropout and `weight_decay` (`0f527fc`), but the model completely collapsed, underfitting and hovering around 8% accuracy (random guessing). We then ripped the regularization out (`9d1f9bd`), but it immediately started heavily overfitting to the training data while validation accuracy flatlined.

## 3. The Pivot to Google Colab
**The Action:** We ported the entire training ecosystem into `colab_train.ipynb`.
**The Rationale:** We desperately needed raw compute power to iterate on our hyperparameters. By shifting to a **Colab T4 GPU**, epoch times plummeted from 300 seconds to just **15 seconds**. This allowed us to rapidly test different configurations in minutes rather than hours.

## 4. The Physics Breakthrough (Beating the Baseline)
**Commits:**
* `a484fda` - Optimize STFT for higher temporal resolution to hit 50% accuracy
* `401950a` - Fix STFT assertion to 32x33 dimensions

**The Narrative & Rationale:**
Even with the Colab GPU, the model hit a hard ceiling of **~41%** accuracy. Your 1D raw IQ baseline was **46%**. A 2D CNN *should* theoretically be better than 1D, so why was it failing?

We traced the bottleneck back to the core physics of the dataset. The initial `precompute_stft.py` used `nperseg=64, noverlap=32`. This squished the 128-sample time domain into just **5 time bins** (creating a 64x5 image). 
* **The Problem:** Modulations like QAM and Phase Shift Keying rely on phase transitions over time. By compressing 128 samples into 5 columns of pixels, the model was practically blind to time. It was trying to classify signals based purely on frequency spikes, which isn't enough information.
* **The Solution:** We altered the physics to `nperseg=32, noverlap=28`. This traded a tiny bit of frequency resolution for massive temporal resolution, resulting in **32x33 spectrograms**. Because our ResNet used an `AdaptiveAvgPool2d` layer, the code dynamically absorbed this new dimension without breaking.

**The Result:** Fed with proper temporal data, the ResNet-18 immediately broke the ceiling, achieving **48.15% validation accuracy** and officially defeating the 1D model.

## 5. Dashboards & Final Cleanup
**Commits:**
* `f333d13` - Add dashboard visualization script
* `5cbe53e` - Sync docstrings and smoke tests to new 32x33 STFT shape
* `5d81028` *(on main)* - Fix CI Pipeline test constraints

**The Narrative & Rationale:**
With the model trained and the weights downloaded, we wrote `dashboard.py` to run localized CPU inference and generate the final visual deliverables (Confusion Matrices, SNR Curves). 

Finally, your teammate noticed the GitHub Actions CI pipeline failing on the new branch. While the neural network didn't care about the dimension change, the hardcoded unit tests were still looking for the old `64x5` shapes. We synchronized the docstrings and test suites to officially establish `32x33` as the new standard, unblocking your teammate and flawlessly merging the branch into `main`.
