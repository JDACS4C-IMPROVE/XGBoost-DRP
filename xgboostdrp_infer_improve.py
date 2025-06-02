import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import xgboost as xgb

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm

# Model-specifc imports
from model_params_def import infer_params # [Req]

filepath = Path(__file__).resolve().parent # [Req]

def extract_subset_fea(df, fea_list: List, fea_sep: str='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]

# [Req]
def run(params: Dict) -> bool:
    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    te_data = pd.read_parquet(Path(params["input_data_dir"]) / test_data_fname)
    fea_list = ["ge", "mordred"]
    fea_sep = "."

    # Test data
    xte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep)
    yte = te_data[[params["y_col_name"]]]
    print("xte:", xte.shape)
    print("yte:", yte.shape)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]
    ) # [Req]

    # Load LightGBM
    model = xgb.XGBRegressor()
    model.load_model(str(modelpath))

    # Compute predictions
    test_pred = model.predict(xte)
    test_true = yte.values.squeeze()

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="xgboostdrp_params.ini",
                                       additional_definitions=infer_params)
    timer_infer = frm.Timer()    
    status = run(params)
    timer_infer.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_infer.json', 
                           extra_dict={"stage": "infer"})
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])