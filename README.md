# model-based-prediction
This repository provides tools for model-based prediction using machine learning techniques. Follow the steps below to set up the environment, train models, and generate figures.

## Install Dependent Python Modules
Ensure you have Python 3 installed. Then, install the required Python libraries by running:
```bash
pip3 install numpy
pip3 install scipy
pip3 install scikit-learn
pip3 install shap
```

## Clone the Repository
Copy the repository from GitHub to your local machine:
```bash
git clone https://github.com/kyesam-jung/model-based-prediction.git
```

## Train Models Using Python Scripts
Navigate to the `codes` directory and use the provided Python scripts to train models and make predictions. Below is a description of the scripts and their usage:

### Classification Models
Run the `classification_nested_cvcr.py` script to train classification models. The script takes two arguments:
1. **Dataset**: Options include `EMP`, `SIM`, or `EMPSIM`.
2. **Dimensionality**: Options include `N/A`, `LowDim`, or `HighDim`.

Commands:
```bash
cd model-based-prediction/codes
python3 classification_nested_cvcr.py EMP N/A
python3 classification_nested_cvcr.py SIM LowDim
python3 classification_nested_cvcr.py EMPSIM LowDim
python3 classification_nested_cvcr.py SIM HighDim
python3 classification_nested_cvcr.py EMPSIM HighDim
```

### Cognition Prediction Models
Use the `prediction_nested_cvcr_cognition.py` script to predict cognitive outcomes. The script takes two arguments:
1. **Dataset**: Options include `EMP`, `SIM`, or `EMPSIM`.
2. **Dimensionality**: Options include `N/A`, `LowDim`, or `HighDim`.

Commands:
```bash
cd model-based-prediction/codes
python3 prediction_nested_cvcr_cognition.py EMP N/A
python3 prediction_nested_cvcr_cognition.py SIM LowDim
python3 prediction_nested_cvcr_cognition.py EMPSIM LowDim
python3 prediction_nested_cvcr_cognition.py SIM HighDim
```

### Personality Prediction Models
The `prediction_nested_cvcr_personality.py` script predicts personality traits. It requires three arguments:
1. **Trait**: Options include `O`, `C`, `E`, `A`, or `N` (representing the Big Five personality traits).
2. **Dataset**: Options include `EMP` or `SIM`.
3. **Dimensionality**: Options include `N/A`, `LowDim`, or `HighDim`.

Commands:
```bash
cd model-based-prediction/codes
python3 prediction_nested_cvcr_personality.py O EMP N/A
python3 prediction_nested_cvcr_personality.py C EMP N/A
python3 prediction_nested_cvcr_personality.py E EMP N/A
python3 prediction_nested_cvcr_personality.py A EMP N/A
python3 prediction_nested_cvcr_personality.py N EMP N/A

python3 prediction_nested_cvcr_personality.py O SIM LowDim
python3 prediction_nested_cvcr_personality.py C SIM LowDim
python3 prediction_nested_cvcr_personality.py E SIM LowDim
python3 prediction_nested_cvcr_personality.py A SIM LowDim
python3 prediction_nested_cvcr_personality.py N SIM LowDim

python3 prediction_nested_cvcr_personality.py O SIM HighDim
python3 prediction_nested_cvcr_personality.py C SIM HighDim
python3 prediction_nested_cvcr_personality.py E SIM HighDim
python3 prediction_nested_cvcr_personality.py A SIM HighDim
python3 prediction_nested_cvcr_personality.py N SIM HighDim
```

## Generate Figures Using MATLAB
To visualize the results, use the MATLAB script provided in the `codes` directory. Run the following commands:
```bash
cd model-based-prediction/codes
generate_figures
```

This will generate the figures used in the study.

## Permutation test
Test null hypothesis of predictive models.

### Classification Models (permutation)
Run the `permutation_classification_nested_cvcr.py` script to train classification models. The script takes two arguments:
1. **Dataset**: Options include `EMP`, `SIM`, or `EMPSIM`.
2. **Dimensionality**: Options include `N/A`, `LowDim`, or `HighDim`.

Commands:
```bash
cd model-based-prediction/codes
python3 permutation_classification_nested_cvcr.py EMP N/A
python3 permutation_classification_nested_cvcr.py SIM LowDim
python3 permutation_classification_nested_cvcr.py EMPSIM LowDim
python3 permutation_classification_nested_cvcr.py SIM HighDim
python3 permutation_classification_nested_cvcr.py EMPSIM HighDim
```

### Cognition Prediction Models (permutation)
Use the `permutation_prediction_nested_cvcr_cognition.py` script to predict cognitive outcomes. The script takes two arguments:
1. **Dataset**: Options include `EMP`, `SIM`, or `EMPSIM`.
2. **Dimensionality**: Options include `N/A`, `LowDim`, or `HighDim`.

Commands:
```bash
cd model-based-prediction/codes
python3 permutation_prediction_nested_cvcr_cognition.py EMP N/A
python3 permutation_prediction_nested_cvcr_cognition.py SIM LowDim
python3 permutation_prediction_nested_cvcr_cognition.py EMPSIM LowDim
python3 permutation_prediction_nested_cvcr_cognition.py SIM HighDim
```

### Personality Prediction Models (permutation)
The `permutation_prediction_nested_cvcr_personality.py` script predicts personality traits. It requires three arguments:
1. **Trait**: Options include `O`, `C`, `E`, `A`, or `N` (representing the Big Five personality traits).
2. **Dataset**: Options include `EMP` or `SIM`.
3. **Dimensionality**: Options include `N/A`, `LowDim`, or `HighDim`.

Commands:
```bash
cd model-based-prediction/codes
python3 permutation_prediction_nested_cvcr_personality.py O EMP N/A
python3 permutation_prediction_nested_cvcr_personality.py C EMP N/A
python3 permutation_prediction_nested_cvcr_personality.py E EMP N/A
python3 permutation_prediction_nested_cvcr_personality.py A EMP N/A
python3 permutation_prediction_nested_cvcr_personality.py N EMP N/A

python3 permutation_prediction_nested_cvcr_personality.py O SIM LowDim
python3 permutation_prediction_nested_cvcr_personality.py C SIM LowDim
python3 permutation_prediction_nested_cvcr_personality.py E SIM LowDim
python3 permutation_prediction_nested_cvcr_personality.py A SIM LowDim
python3 permutation_prediction_nested_cvcr_personality.py N SIM LowDim

python3 permutation_prediction_nested_cvcr_personality.py O SIM HighDim
python3 permutation_prediction_nested_cvcr_personality.py C SIM HighDim
python3 permutation_prediction_nested_cvcr_personality.py E SIM HighDim
python3 permutation_prediction_nested_cvcr_personality.py A SIM HighDim
python3 permutation_prediction_nested_cvcr_personality.py N SIM HighDim
```

## Generate Supplementary Figures Using MATLAB
To visualize the results, use the MATLAB script provided in the `codes` directory. Run the following commands:
```bash
cd model-based-prediction/codes
generate_suppl_figures
```

This will generate the suplementary figures used in the study.

## Notes
- Ensure all dependencies are installed before running the scripts.
- Modify the script arguments as needed to suit your dataset and analysis requirements.
- MATLAB must be installed to generate figures.

For further details, refer to the documentation or contact Kyesam Jung.