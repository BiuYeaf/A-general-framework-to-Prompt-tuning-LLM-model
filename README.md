# Large Language Model Prompt Tuning Framework for classification

Welcome to our Large Language Model Prompt Tuning Framework, a robust solution designed for training large-scale language models efficiently, even on limited resources. This framework leverages the power of Hugging Face's `peft` model and `accelerate` tools, combined with efficient script management through `srun`, providing a streamlined approach for researchers and developers alike.

## Features

- **Resource Efficiency**: Utilize cutting-edge techniques to train large language models with significantly reduced computational requirements.
- **Hugging Face Integration**: Built with Hugging Face's `peft` and `accelerate`, ensuring compatibility with state-of-the-art model training practices.
- **Scalable and Flexible**: Adapt to various scales of resources, from small local setups to large distributed systems.
- **SLURM Support**: Includes scripts for `srun`, making it easy to deploy on SLURM-managed clusters.

## Getting Started

Follow these instructions to set up and run the framework on your system. We recommend you to use python 3.10 version and install the package via

```bash
pip install -r requirements.txt
```

### Prerequisites

Ensure you have the following installed:

- Hugging Face `transformers`
- Hugging Face `accelerator`
- SLURM (for cluster management)

### Usage

1. Run the training script with `srun` if you are using a SLURM cluster, or directly from your terminal:

   ```bash
   sbatch GPTmodel_huggingface_multigpu.sh
   # or
   accelerate launch -m train
   ```

2. We also provide cpu version to eval the model.

3. The code is just for reference, please feel free to modify if you have other needs.