Code must be run from root directory.

Model locations from paper:

Baselines - models\baselines

ANN - models\offset_ahead\cyc

ResNet - models\offset_ahead\res

Random Forest - models\offset_ahead\random_forest

XGBoost - models\offset_ahead\xgboost


Example of naming scheme vs paper (using ANN):

Lagged Feature Set | File Name

------------------------------

Daily Velocity | cyc_ahead_model_wide_noOffset.py 

Daily and Hourly Velocity | cyc_ahead_model_wide_noOffset_4hours.py 

Previous Day's Weather | cyc_ahead_model_wide_24time.py

Previous Day's GHI | cyc_ahead_model_wide_24ghi.py
