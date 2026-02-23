# Define util functions for Databricks

from datetime import datetime
print(datetime.now().strftime('%Y%m%d%H%M'))
print('#'*20 + datetime.now().strftime(' %Y.%m.%d ') + '#'*20)

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
from joblib import Parallel, delayed
import statsmodels.api as sm

import pyarrow.parquet as pq
import s3fs
import math
import yaml
import matplotlib.pyplot as plt
from matplotlib import font_manager
import datetime as dt
import pyarrow.parquet as pq
import gc
import matplotlib.cm

import sys
sys.path.append('../../')

# loaded data if required
fs = s3fs.S3FileSystem()

data_path = "s3://"

### analysis and check issue
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

def save2local_v0(df, df_name, path = data_path, f_type = 'csv'):
    if f_type == 'csv':
        df.to_csv(path + df_name + '.csv',index=False)
        print("Saved to local: {}.csv, at {}".format(df_name, path))
    else:
        df.to_parquet(path + '{}.parquet'.format(df_name),index=False)
        print("Saved to local: {}.parquet, at {}".format(df_name, path))

def save2local(df, df_name, path = data_path, f_type = 'csv', num_files=1, **parquet_wargs):
    if num_files == 1:
        save2local_v0(df, df_name, path = path, f_type = f_type)
    else:
        chunk_size = len(df) // num_files + 1
        for i in range(0, len(df), chunk_size):
            slc = df.iloc[i : i + chunk_size]
            chunk = int(i/chunk_size)
            fname = f"{path}{df_name}_{chunk:04d}"
            if f_type == 'csv':
                slc.to_csv(f'{fname}.csv', index=False)
                print(f"Saved to local: {fname}.csv")
            else:
                slc.to_parquet(f'{fname}.parquet', engine="pyarrow", index=False, **parquet_wargs)
                print(f"Saved to local: {fname}.parquet")


def read_local_file(df_name, path = data_path, sep=','):
    return pd.read_csv(path + df_name + '.csv', sep=sep)


def to_uri(bucket, key):
    return f's3://{bucket}/{key}'

def read_multipart_parquet_s3_bucket(bucket, dir_path = '', prefix_filename='', postfix_filename=''):
    fs = s3fs.S3FileSystem()
    paths = [path for path in fs.ls(to_uri(bucket, dir_path)) if path.startswith(f'{bucket}/{dir_path}' + prefix_filename) and path.endswith(postfix_filename)]
    return pq.ParquetDataset(paths, filesystem=fs).read().to_pandas(date_as_object=False)


def read_multipart_parquet_s3(dir_path_s3 = '', prefix_filename='', postfix_filename=''):
    """
    """
    fs = s3fs.S3FileSystem()
    paths = [path for path in fs.ls(dir_path_s3) if path.startswith(dir_path_s3[5:] + prefix_filename) and path.endswith(postfix_filename)]
    op = pd.DataFrame()
    for _p in paths:
        _op = pq.ParquetDataset(_p, filesystem=fs).read().to_pandas(date_as_object=False)
        op = pd.concat([op, _op])
    return op

 
def date_to_week_id(date):
    assert isinstance(date, (str, pd.Timestamp, pd.Series, dt.date, pd.core.indexes.datetimes.DatetimeIndex))
    if isinstance(date, (str, pd.Timestamp, dt.date)):
        date = pd.Timestamp(date)
        if date.dayofweek == 6:  # If sunday, replace by next monday to get the correct iso week
            date = date + pd.Timedelta(1, unit="D")
        week_id = int(str(date.isocalendar()[0]) + str(date.isocalendar()[1]).zfill(2))
        return week_id
    else:
        df = pd.DataFrame({"date": pd.to_datetime(date)})
        df["dow"] = df["date"].dt.dayofweek
        df.loc[df["dow"] == 6, "date"] = df.loc[df["dow"] == 6, "date"] + pd.Timedelta(1, unit="D")
        df["week_id"] = df["date"].apply(lambda x: int(str(x.isocalendar()[0]) + str(x.isocalendar()[1]).zfill(2)))
        return df["week_id"].tolist()

def week_id_to_date(week_id):
    assert isinstance(week_id, (int, np.integer, pd.Series, list))
    if isinstance(week_id, (int, np.integer)):
        return pd.to_datetime(str(week_id) + "-0", format="%G%V-%w") - pd.Timedelta(1, unit="W")
    else:
        if isinstance(week_id, (list)):
            week_id = pd.Series(week_id)
        return pd.to_datetime(week_id.astype(str) + "-0", format="%G%V-%w") - pd.Timedelta(1, unit="W")

def apply_delta_week(cutoff_date, delta_week):
    date_start = week_id_to_date(cutoff_date)
    delta_date = date_start - dt.timedelta(weeks=delta_week)
    return delta_date


def read_multipart_parquet_local(data_path = '', prefix_filename='', postfix_filename='.parquet'):
    if os.path.isdir(data_path):
        paths = [data_path + path for path in os.listdir(data_path) if path.startswith(prefix_filename) and path.endswith(postfix_filename)]
        return pq.ParquetDataset(paths).read().to_pandas(date_as_object=False)
    else:
        return pq.ParquetDataset(data_path).read().to_pandas(date_as_object=False)

print("Import DB_common_utils successfully!")

