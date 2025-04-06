# LIBWO

---

## Project Structure

```bash
main/
│
├── cec/                       # Benchmark testing
│   ├── cec2017/               # CEC2017 test suite
│   │   ├── cec2017_run/       # Main execution logic and result output
│   │   ├── BWO/               # Black Widow Optimization implementation
│   │   └── LIBWO/             # Lagrange-Interpolated Black Widow Optimization
│   └── cec2022/               # (Optional future support)
│
├── ResNet/                    # Deep learning optimization & training
│   ├── model/                 # ResNet18 architecture
│   ├── ModelTrain/            # Main training logic with LIBWO/BWO
│   ├── BWO/                   # BWO for model hyperparameter tuning
│   └── LIBWO/                 # LIBWO for hyperparameter tuning
│
└── videos/
    └── cec_run.mp4            # Demo of CEC benchmark execution and result visualization
```

---

## Module Description

### Benchmark Optimization (`cec`)

- [`cec/cec2017/cec2017_run`](./cec/cec2017/cec2017_run)  
  Main execution module: calls CEC2017 test functions and evaluates baseline algorithms (such as BWO and LIBWO), outputs results.

- [`cec/cec2017/BWO`](./cec/cec2017/BWO)  
  Source code implementation of the BWO algorithm for benchmark optimization tasks.

- [`cec/cec2017/LIBWO`](./cec/cec2017/LIBWO)  
  Core of this project: **Lagrange-Interpolated Black Widow Optimization**, enhancing search ability and convergence.

### Deep Learning Optimization (ResNet)

- [`ResNet/model`](./ResNet/model)  
  Definition of the ResNet18 model, built with TensorFlow/Keras.

- [`ResNet/ModelTrain`](./ResNet/ModelTrain)  
  Main training workflow: loads data, invokes BWO/LIBWO for learning rate optimization, trains and evaluates the model.

- [`ResNet/BWO`](./ResNet/BWO)  
  BWO algorithm used for tuning hyperparameters such as learning rate.

- [`ResNet/LIBWO`](./ResNet/LIBWO)  
  Improved BWO using Lagrange interpolation to optimize model performance.

### Demo Video

- [`videos/cec_run.mp4`](./videos/cec_run.mp4)  
  Demonstrates the execution process and visualization of CEC benchmark test results.

---

## ⚙️ Environment Setup

It is recommended to use [Anaconda](https://www.anaconda.com/) to manage the environment. This project is based on **Python 3.12**, with the following configuration:

| Library         | Version     |
|----------------|-------------|
| Python          | 3.12        |
| CUDA            | 11.2        |
| cuDNN           | 8.2         |
| TensorFlow-GPU  | 2.18        |
| Keras           | 2.12        |
| Scikit-learn    | 1.5.1       |
