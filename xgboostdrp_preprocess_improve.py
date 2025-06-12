import sys
from pathlib import Path
import pandas as pd

# Core improvelib imports
import improvelib.utils as frm
# Application-specific (DRP) imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Model-specifc imports
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent


# [Req]
def run(params):
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    # not needed for this data/model

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    print("Load omics data.")
    omics = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])
    omics_transform = params['cell_transcriptomic_transform']

    print("Load drug data.")
    drugs = frm.get_x_data(file = params['drug_mordred_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    drugs_transform = params['drug_mordred_transform']

    print("Load train response data.")
    response_train = frm.get_response_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   response_file=params['y_data_file'])
    
    print("Find intersection of training data.")
    response_train = frm.get_response_with_features(response_train, omics, params['canc_col_name'])
    response_train = frm.get_response_with_features(response_train, drugs, params['drug_col_name'])
    omics_train = frm.get_features_in_response(omics, response_train, params['canc_col_name'])
    drugs_train = frm.get_features_in_response(drugs, response_train, params['drug_col_name'])

    print("Determine transformations.")
    frm.determine_transform(omics_train, 'omics_transform', omics_transform, params['output_dir'])
    frm.determine_transform(drugs_train, 'drugs_transform', drugs_transform, params['output_dir'])

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # Dict with split files corresponding to the three sets (train, val, and test)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}

    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = frm.get_response_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                response_file=params['y_data_file'])
        response_stage = frm.get_response_with_features(response_stage, omics, params['canc_col_name'])
        response_stage = frm.get_response_with_features(response_stage, drugs, params['drug_col_name'])
        omics_stage = frm.get_features_in_response(omics, response_stage, params['canc_col_name'])
        drugs_stage = frm.get_features_in_response(drugs, response_stage, params['drug_col_name'])

        print(f"Transform {stage} data.")
        omics_stage = frm.transform_data(omics_stage, 'omics_transform', params['output_dir'])
        drugs_stage = frm.transform_data(drugs_stage, 'drugs_transform', params['output_dir'])

        # [Req] Build data name
        data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage=stage)

        print(f"Merge {stage} data")
        y_df_cols = response_stage.columns.tolist()
        data = response_stage.merge(omics_stage, on=params["canc_col_name"], how="inner")
        data = data.merge(drugs_stage, on=params["drug_col_name"], how="inner")
        data = data.sample(frac=1.0).reset_index(drop=True) # shuffle

        print(f"Save {stage} data")
        xdf = data.drop(columns=y_df_cols)
        xdf[params['y_col_name']] = data[params['y_col_name']]
        xdf.to_parquet(Path(params["output_dir"]) / data_fname) # saves ML data file to parquet
        
        # [Req] Save y dataframe for the current stage
        ydf = data[y_df_cols]
        frm.save_stage_ydf(ydf, stage, params["output_dir"])

    return params["output_dir"]


# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="xgboostdrp_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])