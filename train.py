import pickle

import pandas as pd
import numpy as np
import sklearn

from xgboost import XGBClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

def load_data():
    df = pd.read_csv('Thyroid_Diff.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.rename(columns={'t': 'tumor', 'n': 'node', 'm': 'metastasis'})

    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    df.recurred = (df.recurred == 'yes').astype(int)

    df = df.drop(columns=['risk'], errors='ignore')

    return df


def train_model(df):
    numerical = ['age']
    categorical = ['gender', 'smoking', 'hx_smoking', 'hx_radiothreapy', 
                   'thyroid_function', 'physical_examination', 'adenopathy','pathology', 
                   'focality', 'tumor', 'node', 'metastasis', 'stage', 'response']
    
    y_train = df.recurred
    train_dict = df[categorical + numerical].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(sparse=True),
            XGBClassifier(
            learning_rate=0.1,     # eta
            min_child_weight=1,
            max_depth=6,
            n_estimators=300,      # or whatever you used
            subsample=1.0,         # keep defaults unless tuned
            colsample_bytree=1.0,
            eval_metric="logloss"  # important for sklearn API
        )
    )

    pipeline.fit(train_dict, y_train)

    return pipeline


def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    
    print(f'Model saved to {output_file}')



df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')