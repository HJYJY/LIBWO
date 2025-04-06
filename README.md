# LIBWO

---

## Project Structure

```
main/
│
├── cec/                       # Benchmark testing
│   ├── cec2017/               # CEC2017 test suite
│   │   ├── cec2017_run/       # Main run logic and output results
│   │   ├── BWO/               # Black Widow Optimization implementation
│   │   └── LIBWO/             # Lagrange-Interpolated Black Widow Optimization
│   └── cec2022/               # CEC2022 test suite (future support)
│       ├── cec2022_run/       # Main run logic and output results for CEC2022
│       ├── BWO/               # Black Widow Optimization implementation for CEC2022
│       └── LIBWO/             # Lagrange-Interpolated Black Widow Optimization for CEC2022
│
├── ResNet/                    # Deep learning optimization & training
│   ├── model/                 # ResNet18 architecture
│   ├── ModelTrain/            # Main training logic with LIBWO/BWO
│   ├── BWO/                   # BWO applied to model hyperparameter tuning
│   └── LIBWO/                 # LIBWO applied to hyperparameter tuning
│
└── videos/
    └── cec_run.mp4            # Demo of CEC benchmark test and output
```

---

## Module Descriptions

### Benchmark Optimization (`cec`)

- [`cec/cec2017/cec2017_run`](./cec/cec2017/cec2017_run)  
  Main execution module: runs CEC2017 test functions, evaluates baseline algorithms (e.g., BWO and LIBWO), and outputs results.

- [`cec/cec2017/BWO`](./cec/cec2017/BWO)  
  Source code for the **Black Widow Optimization (BWO)** algorithm used for benchmark tasks.

- [`cec/cec2017/LIBWO`](./cec/cec2017/LIBWO)  
  The core component of this project: **Lagrange Interpolation-enhanced Black Widow Optimization (LIBWO)** algorithm, boosting search ability and convergence.

- [`cec/cec2022/cec2022_run`](./cec/cec2022/cec2022_run)  
  Main execution module for CEC2022: runs CEC2022 test functions, evaluates baseline algorithms (e.g., BWO and LIBWO), and outputs results.

- [`cec/cec2022/BWO`](./cec/cec2022/BWO)  
  Source code for the **Black Widow Optimization (BWO)** algorithm applied to CEC2022 benchmark tasks.

- [`cec/cec2022/LIBWO`](./cec/cec2022/LIBWO)  
  The **Lagrange Interpolation-enhanced Black Widow Optimization (LIBWO)** algorithm applied to CEC2022. Enhances the BWO search process for CEC2022 benchmark functions.

### Deep Learning Optimization (`ResNet`)

- [`ResNet/model`](./ResNet/model)  
  Definition of the **ResNet18** model, implemented using TensorFlow/Keras.

- [`ResNet/ModelTrain`](./ResNet/ModelTrain)  
  Main training pipeline: loads data, applies BWO/LIBWO for learning rate optimization, trains and evaluates the model.

  **Model Training Process**  
  After downloading and configuring all datasets correctly, ensure the runtime environment is properly set up. Then, update the dataset paths in the `ModelTrain.py` file to your local directories. Run the script to start the full training pipeline.

  Initially, the LIBWO algorithm is called for hyperparameter tuning. Through iterative global search, LIBWO finds the optimal hyperparameter combinations and their corresponding fitness values.

  Once the optimal values are found, the system automatically applies them to the ResNet18 model, which then starts training until convergence is achieved—ensuring maximum performance.

  After training, the system evaluates the model and outputs key metrics such as test accuracy, test loss, and additional indicators including precision, recall, and F1-score. This end-to-end workflow ensures efficient training and evaluation, resulting in optimal classification performance.

- [`ResNet/BWO`](./ResNet/BWO)  
  BWO algorithm applied to model hyperparameter tuning (e.g., learning rate).

- [`ResNet/LIBWO`](./ResNet/LIBWO)  
  Enhanced BWO variant using Lagrange Interpolation to further improve model performance.

### Demo Video

- [`videos/cec_run.mp4`](./videos/cec_run.mp4)  
  Demonstrates the execution and result visualization of the CEC benchmark test functions.

---

## Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) to manage your Python environment. This project is based on **Python 3.12** with the following recommended configuration:

| Library         | Version     |
|----------------|-------------|
| Python          | 3.12        |
| CUDA            | 11.2        |
| cuDNN           | 8.2         |
| TensorFlow-GPU  | 2.18        |
| Keras           | 2.12        |
| Scikit-learn    | 1.5.1       |
