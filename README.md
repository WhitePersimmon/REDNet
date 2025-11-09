# REDNet
Official PyTorch implementation of REDNet, a robust time-series forecasting model for edge devices via two-stage knowledge distillation.

## Highlights
* **Problem:** Large models are robust but too large for edge devices; lightweight models are efficient but 'fragile' and fail on real-world noisy data.
* **Solution (REDNet):** We propose a two-stage knowledge distillation framework to transfer the deep temporal understanding and generalization strength from a large 'teacher' model to a compact 'student' model.
* **Core Innovations:**
    1.  A novel **distillation-oriented student architecture** (MLP-BiGRU) designed to efficiently absorb temporal knowledge.
    2.  A **two-stage distillation process** (feature alignment + end-to-end fine-tuning) for robust adaptation.
* **Results:** REDNet surpasses SOTA lightweight models in accuracy and, crucially, **inherits the teacher's robustness** against severe data corruption and concept drift, making it a practical solution for reliable edge forecasting.

## How to Run

### Acknowledgement & Setup

First and foremost, we want to thank the authors of the **Tiny Time Mixers (TTMs)**. Our work and benchmarking framework are built directly upon their excellent [official repository]([https://github.com/google-research/tiny-time-mixers](https://github.com/ibm-granite/granite-tsfm)).

To run our experiments, you must first clone the `tinytimemixer` repository and place our provided files in the correct directories.

### Running the Full Benchmark

Once the setup is complete, our full benchmarking and model evaluation can be run from the following location:
`notebooks/hfdemo/tinytimemixer/full_benchmarking`
