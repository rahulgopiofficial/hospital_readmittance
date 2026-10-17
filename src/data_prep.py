"""
data_prep.py — Stage 2: data cleaning, feature engineering, leakage-safe
preprocessing and splitting for the Hospital Readmission pipeline.

Implement the functions below; they are imported by Data_Preparation.ipynb and by
src/train.py. Expected end state: a model frame of ~69,973 rows at ~9% positive, and a
stratified train/val/test split with the preprocessor fit on the TRAIN split only.
"""
from matplotlib import axis
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config as cfg

ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def load_raw():
    """TODO: read cfg.RAW_CSV with na_values=['?'] so '?' becomes NaN; return the DataFrame."""
    df = pd.read_csv(cfg.RAW_CSV, na_values=['?'])
    return df


def clean(df):
    """TODO (Stage 2.1): keep the FIRST encounter per patient (drop_duplicates('patient_nbr'))
    to prevent patient-level leakage; drop expired/hospice discharges
    (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS); build the binary target
    cfg.TARGET = 1 if readmitted == '<30' else 0; drop the cfg.DROP_COLS that are present.
    Return the cleaned frame (~69,973 rows, ~9% positive)."""

    # Notebook Stage 2.1.1 also asks to drop constant/zero-variance columns 
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_cols, errors='ignore', inplace=True)

    #keep the FIRST encounter per patient (drop_duplicates('patient_nbr')) to prevent patient-level leakage
    df.drop_duplicates(subset=['patient_nbr'], keep='first', inplace=True, ignore_index=True)

    # drop expired/hospice discharges (discharge_disposition_id in cfg.EXPIRED_HOSPICE_DISPOSITIONS)
    df = df[~df['discharge_disposition_id'].isin(cfg.EXPIRED_HOSPICE_DISPOSITIONS)]

    # build the binary target cfg.TARGET = 1 if readmitted == '<30' else 0
    df[cfg.TARGET] = (df['readmitted'] == '<30').astype(int)

    # drop the cfg.DROP_COLS that are present
    df.drop(columns=cfg.DROP_COLS, inplace=True, errors='ignore')

    # Drop the original 'readmitted' column to prevent data leakage
    df.drop(columns=['readmitted'], errors='ignore', inplace=True)

    return df


def engineer_features(df):
    """TODO (Stage 2.2): group ICD9 diag_1/2/3 into clinical categories; map the age band to a
    numeric midpoint; service_utilization = number_outpatient + number_emergency + number_inpatient;
    num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS; collapse high-cardinality
    medical_specialty to top-k + 'Other'. Return the engineered frame."""
    
    # group ICD9 diag_1/2/3 into clinical categories;
    # 1. Define the ICD-9 mapping logic
    def map_icd9(code):
        # Handle missing values
        if pd.isna(code) or code == '?':
            return 'Other'
            
        code_str = str(code)
        
        # ICD-9 codes starting with V or E are supplemental/external; map to Other
        if code_str.startswith('V') or code_str.startswith('E'):
            return 'Other'
            
        try:
            # Convert to float to handle sub-categories like '250.01'
            num_code = float(code_str)
        except ValueError:
            return 'Other'
            
        # Apply the justified clinical groupings
        if 250 <= num_code < 251:
            return 'Diabetes'
        elif (390 <= num_code <= 459) or num_code == 785:
            return 'Circulatory'
        elif (460 <= num_code <= 519) or num_code == 786:
            return 'Respiratory'
        elif 800 <= num_code <= 999:
            return 'Injury'
        else:
            return 'Other'
            
    # 2. Apply the mapping to all three diagnosis columns
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(map_icd9)

    # map the age band to a numeric midpoint; 
    age_mapping = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    
    if 'age' in df.columns:
        df['age'] = df['age'].map(age_mapping)

    # service_utilization = number_outpatient + number_emergency + number_inpatient;
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

    # num_med_changes = count of 'Up'/'Down' across cfg.MED_COLS; 
    df['num_med_changes'] = df[cfg.MED_COLS].isin(['Up', 'Down']).sum(axis=1)

    # collapse high-cardinality medical_specialty to top-k + 'Other'
    if 'medical_specialty' in df.columns:
        # Find the top 10 most frequent medical specialties
        top_10 = df['medical_specialty'].value_counts().nlargest(10).index
        
        # Keep the value if it is in the top 10, otherwise replace it with 'Other'
        df['medical_specialty'] = df['medical_specialty'].where(df['medical_specialty'].isin(top_10), 'Other')

    return df


def get_feature_lists(df):
    """TODO: return (numeric_cols, categorical_cols), excluding cfg.TARGET."""

    # Numeric columns list
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Categorical columns list
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # If target column present need to remove it
    if cfg.TARGET in numeric_cols:
        numeric_cols.remove(cfg.TARGET)
    if cfg.TARGET in categorical_cols:
        categorical_cols.remove(cfg.TARGET)

    return numeric_cols, categorical_cols


def build_preprocessor(numeric, categorical):
    """TODO (Stage 2.3): return ONE ColumnTransformer — numeric: median SimpleImputer + StandardScaler;
    categorical: constant SimpleImputer + OneHotEncoder(handle_unknown='ignore'). Do NOT fit it here
    (it is fit on the TRAIN split only)."""

    # 1. Blueprint for numerical columns: Fill missing with median, then scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # 2. Blueprint for categorical columns: Fill missing with a constant, then One-Hot Encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Combine both blueprints into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric),
            ('cat', cat_pipeline, categorical)
        ]
    )
    
    return preprocessor


def get_model_frame():
    """TODO: return engineer_features(clean(load_raw()))."""
    return engineer_features(clean(load_raw()))


def get_splits(df=None):
    """TODO (Stage 2.4): create a STRATIFIED train/val/test split using cfg.RANDOM_STATE,
    cfg.TEST_SIZE and cfg.VAL_SIZE (val taken from the train split). Return
    X_train, X_val, X_test, y_train, y_val, y_test. The class ratio must be preserved across splits;
    the preprocessor is fit on TRAIN only (in the notebook / train.py), never on the full data."""

    # If the user didn't provide a DataFrame, this is because df=None was mentioned
    if df is None:
        df = engineer_features(clean(load_raw()))

    # 1. Separate features (X) from the target (y)
    X = df.drop(columns=[cfg.TARGET])
    y = df[cfg.TARGET]

    # 2. First Split: Pull out the Test set (using cfg variables)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=cfg.TEST_SIZE, 
        stratify=y, 
        random_state=cfg.RANDOM_STATE
    )

    # 3. Second Split: Separate the temporary mix into Train and Val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=cfg.VAL_SIZE, 
        stratify=y_temp, 
        random_state=cfg.RANDOM_STATE
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test