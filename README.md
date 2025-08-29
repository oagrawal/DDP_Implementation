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

## Performance Analysis and Takeaways

1.  The average time per epoch for single-node training was 12.302 seconds. For Distributed Data Parallel (DDP) training, the average times were as follows: 9.045 seconds per epoch with 2 GPUs and 5.894 seconds per epoch with 4 GPUs. This means that using 2 GPUs reduced the training time by 3.257 seconds per epoch compared to single-node training, while using 4 GPUs reduced the training time by 6.408 seconds per epoch compared to the single-node setup and 3.151 seconds compared to the 2-GPU training. Thus, using distributed accelerators significantly speeds up the training process per epoch.

2.  Increasing the number of GPUs from 2 to 4 decreased the training time by 3.151 seconds, which is approximately a 34% reduction when doubling the number of GPUs. While this is a significant speedup, the performance scaling is not linear. This is evident in our data, as the time saved when increasing the GPUs from 1 to 2 is roughly the same as when increasing from 2 to 4. One reason for this is the increased load from synchronizing gradients across multiple GPUs. Specifically, the NCCL library uses operations such as AllReduce, which leads to communication overhead. Since communication is relatively slower than computation, it becomes a bottleneck that worsens as more GPUs are added. This explains why the performance gains are not linear.

3.  Initially, when we used DDP to train and test the neural network, the accuracy was not consistent with the results from single-node training. After debugging, we discovered that the test function was not testing properly across multiple GPUS. To resolve this, we decided to use a single GPU for testing the neural network, which corrected the issue.

Additionally, as we scale training with more GPUs, the communication overhead due to gradient synchronization becomes a significant factor, as mentioned earlier. With more GPUs, the communication overhead increases, reducing speed gains. Furthermore, the likelihood of one machine stalling or lagging behind others increases, creating another bottleneck in the training process and a limitation of DDP as we scale to more machines.

For distributing work, we met either in person or online and collaborated on the project together.