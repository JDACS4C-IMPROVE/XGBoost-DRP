# XGBoost-DRP

---

This repository demonstrates how to use the [IMPROVE library](https://jdacs4c-improve.github.io/docs/) for building a drug response prediction model using XGBoost.


## Dependencies
Installation instructions are detailed below in [Step-by-step instructions](#step-by-step-instructions).


ML framework:
+ [SciKit-Learn](https://scikit-learn.org/)

IMPROVE dependencies:
+ [IMPROVE](https://github.com/JDACS4C-IMPROVE/IMPROVE)

## Dataset
Benchmark data for Drug Response Prediction can be downloaded from this [site](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/drp_data_v0.2.0).



# Step-by-step instructions

### 1. Clone the model repository and checkout the develop branch (or tag of your choice)
```bash
git clone https://github.com/JDACS4C-IMPROVE/XGBoost-DRP
cd XGBoost-DRP
git checkout develop
```


### 2. Set computational environment
Create conda env using `yml`
```bash
conda env create -f xgboostdrp_environment.yml
```

or

```bash
conda create -n xgb python=3.7 pandas=1.3.5 scikit-learn=1.0.2 pyyaml=6.0 pyarrow=9.0.0
pip install xgboost==1.6.2
```


### 3. Preprocess benchmark data to construct model input data 
```bash
python xgboostdrp_preprocess_improve.py --input_dir ./drp_data_v0.2.0 --output_dir exp_result
```

Preprocesses the data and creates train, validation (val), and test datasets.

Generates:
* three model input data files
* three tabular data files, each containing the synergy values and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`



### 4. Train model
```bash
python xgboostdrp_train_improve.py --input_dir exp_result --output_dir exp_result
```

Trains a model using the model input data.

Generates:
* trained model
* predictions on val data (tabular data): `val_y_data_predicted.csv`
* prediction performance scores on val data: `val_scores.json`


### 5. Run inference on test data with the trained model
```bash
python xgboostdrp_infer_improve.py --input_data_dir exp_result --input_model_dir exp_result --output_dir exp_result --calc_infer_score true
```

Evaluates the performance on a test dataset with the trained model.

Generates:
* predictions on test data (tabular data): `test_y_data_predicted.csv`
* prediction performance scores on test data: `test_scores.json`

