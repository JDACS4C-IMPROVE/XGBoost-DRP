import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Model-specifc imports
from model_params_def import train_params # [Req]

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params: Dict) -> Dict:
    # ------------------------------------------------------
    # [Req] Build model path
    # ------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    train_data = pd.read_parquet(Path(params["input_dir"]) / train_data_fname)
    val_data = pd.read_parquet(Path(params["input_dir"]) / val_data_fname)

    # Train data
    ytr = train_data[[params["y_col_name"]]]
    xtr = train_data.drop(columns=[params['y_col_name']])
    print("xtr:", xtr.shape)
    print("ytr:", ytr.shape)

    # Val data
    yvl = val_data[[params["y_col_name"]]]
    xvl = val_data.drop(columns=[params['y_col_name']])
    print("xvl:", xvl.shape)
    print("yvl:", yvl.shape)

    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------
    xgb_args = {'learning_rate': params['learning_rate'],
                'early_stopping_rounds': params['patience'],
                'max_depth': params['max_depth'],
                'min_child_weight': params['min_child_weight'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'gamma': params['gamma'],
                'lambda': params['lambda'],
                'alpha': params['alpha'],
                }
    
    model = xgb.XGBRegressor(objective='reg:squarederror', **xgb_args)
    model.fit(xtr, ytr, eval_set=[(xvl, yvl)])


    model.save_model(str(modelpath))
    del model

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    model = xgb.XGBRegressor()
    model.load_model(str(modelpath))

    # Compute predictions
    val_pred = model.predict(xvl)
    val_true = yvl.values.squeeze()
   
     # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="xgboostdrp_params.ini",
                                       additional_definitions=train_params)
    timer_train = frm.Timer()    
    val_scores = run(params)
    timer_train.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_train.json', 
                           extra_dict={"stage": "train"})
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])