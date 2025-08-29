# PyTorch DDP on AWS SageMaker Demo

This repository provides a demonstration of how to use PyTorch's Distributed Data Parallel (DDP) to train a VGG11 model on the CIFAR-10 dataset. The code is structured to be easily run on AWS SageMaker for multi-GPU training.

## Project Structure

```
.
├── src/
│   ├── model.py            # VGG11 model definition
│   ├── trainer.py          # Training and testing loop functions
│   └── train.py            # Main training script (entry point)
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

- **VGG11 Model**: A VGG11 implementation for CIFAR-10 classification.
- **Distributed Training**: Utilizes PyTorch's `DistributedDataParallel` for efficient multi-GPU training.
- **SageMaker Ready**: The main training script `src/train.py` is designed to work seamlessly with AWS SageMaker's PyTorch estimator, which automatically sets the necessary environment variables for distributed training.
- **Standalone Execution**: The script can also be run locally for single-GPU training or multi-GPU training using `torchrun`.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd ddp-sagemaker-demo
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Local Training

#### Single GPU

To run the training on a single GPU (or CPU if no CUDA is available), simply execute the `train.py` script:

```bash
python src/train.py --epochs 5 --batch_size 256
```

#### Multi-GPU with `torchrun`

If you have multiple GPUs on your local machine, you can launch a distributed training job using `torchrun`. `torchrun` will manage the creation of multiple processes and set the required environment variables (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`).

For example, to train on 2 GPUs:

```bash
torchrun --nproc_per_node=2 src/train.py --epochs 15 --batch_size 256
```

### Training on AWS SageMaker

This code is designed to be used with the SageMaker PyTorch estimator.

1.  **Upload data to S3 (if needed).** The script will download the CIFAR-10 dataset automatically, but for custom datasets, you would typically use an S3 data source.

2.  **Create a SageMaker PyTorch Estimator.** In your SageMaker Notebook or Python script, you would define an estimator like this:

    ```python
    from sagemaker.pytorch import PyTorch

    estimator = PyTorch(
        entry_point='train.py',
        source_dir='./src',
        role='<Your-SageMaker-IAM-Role>',
        instance_count=1,
        instance_type='ml.p3.8xlarge', # Example: instance with 4 GPUs
        framework_version='1.13', # Specify your PyTorch version
        py_version='py39',
        hyperparameters={
            'epochs': 15,
            'batch_size': 256,
            'lr': 0.1
        },
        # For DDP, specify the distribution strategy
        distribution={'pytorchddp': {'enabled': True}}
    )

    estimator.fit()
    ```

When you call `.fit()`, SageMaker will:

- Provision the specified number of instances (`instance_count`).
- Copy your code from `source_dir` to each instance.
- Set up the distributed environment.
- Run your `train.py` script on each of the 4 GPUs of the `ml.p3.8xlarge` instance.

This setup demonstrates a clean, professional approach to organizing a PyTorch DDP project for both local development and cloud deployment on SageMaker.
