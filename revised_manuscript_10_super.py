import calendar
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.dates as mdates
from scipy.stats import kendalltau
import statsmodels.api as sm
from datetime import datetime, timedelta
import copy
from matplotlib.ticker import MultipleLocator, FuncFormatter, MaxNLocator, AutoMinorLocator, NullLocator
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu
import ruptures as rpt
from scipy.stats import linregress
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from sklearn.utils import resample
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
import collections
from scipy.stats import kendalltau, theilslopes, norm
from statsmodels.tsa.stattools import acf
import math as mt
import re
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

sns.set_style('whitegrid')
plt.close('all')
plt.rcdefaults()

def convert_timestamp(year, month, day, hour, minute):
    hour = int(hour) % 24
    minute = int(minute) % 60
    return f'{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:00'

def process_year_data(directory_path, year):
    file_names = [file for file in os.listdir(directory_path) if file.endswith('.FLW')]

    data = pd.DataFrame(columns = ['date_and_time', 'dmps_file_relative_humidity', 'dmps_flw_file_temperature'])

    for file_name in file_names:
        month = int(file_name[4:6])
        day = int(file_name[6:8])

        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path, delim_whitespace = True, usecols = [0, 10, 11], header = None)

        df[0] = df.iloc[:, 0].apply(lambda x: convert_timestamp(year, month, day, int(x), str(x - int(x))[2:]))

        df.columns = ['date_and_time', 'dmps_file_relative_humidity', 'dmps_flw_file_temperature']

        data = pd.concat([data, df], ignore_index = True)

    data['date_and_time'] = pd.to_datetime(data['date_and_time'])

    return data

base_directory = r'./data'
years = [2018, 2019, 2020, 2021, 2022]

all_data = []
for year in years:
    year_directory = os.path.join(base_directory, f'Particle{str(year)[2:]}')
    year_data = process_year_data(year_directory, year)
    all_data.append(year_data)

combined_data = pd.concat(all_data, ignore_index = True)
combined_data = combined_data.iloc[1:].reset_index(drop=True)

pm_both_sca_2018 = pd.read_csv('./data/smr_20180101.csv')

pm_both_sca_2018 = pm_both_sca_2018.drop(index=pm_both_sca_2018.index[1:131035])
pm_both_sca_2018 = pm_both_sca_2018.reset_index(drop=True)

pm_both_sca_2022 = pd.read_csv("./data/smr_20220101.csv")


def extract_pm_data(df_year):
    df_year = df_year.iloc[1:]

    df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'] = pd.to_datetime(
        df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'],
        format='%Y-%m-%d %H:%M:%S'
    )

    pm_1_range = [(5, 9), (25, 29), (45, 49)]
    pm_10_range = [(15, 19), (35, 39), (55, 59)]

    pm_1_minutes = [minute for start, end in pm_1_range for minute in range(start, end + 1)]
    pm_10_minutes = [minute for start, end in pm_10_range for minute in range(start, end + 1)]

    pm_1_data = df_year[df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'].dt.minute.isin(pm_1_minutes)]
    pm_10_data = df_year[df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'].dt.minute.isin(pm_10_minutes)]

    return pm_1_data, pm_10_data

pm_1_sca_2018, pm_10_sca_2018 = extract_pm_data(pm_both_sca_2018)
pm_1_sca_2022, pm_10_sca_2022 = extract_pm_data(pm_both_sca_2022)

pm_1_2018_columns = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Semicolon delimited list of cut sizes present': 'pm_cut_size_range',
    'total light scattering coefficient at 450 nm (Mm-1)': 'pm_1_sca_450',
    'total light scattering coefficient at 550 nm (Mm-1)': 'pm_1_sca_550',
    'total light scattering coefficient at 700 nm (Mm-1)': 'pm_1_sca_700',
    'backwards hemispheric light scattering coefficient at 450 nm (Mm-1)': 'pm_1_bsca_450',
    'backwards hemispheric light scattering coefficient at 550 nm (Mm-1)': 'pm_1_bsca_550',
    'backwards hemispheric light scattering coefficient at 700 nm (Mm-1)': 'pm_1_bsca_700',
    'measurement cell temperature (°C)': 'pm_1_sample_temperature',
    'measurement cell relative humidity (%)': 'pm_1_relative_humidity',
    'measurement cell pressure (hPa)': 'pm_1_sample_pressure'
    }

pm_1_2022_columns = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Semicolon delimited list of cut sizes present': 'pm_cut_size_range',
    'Aerosol light scattering coefficient (Mm⁻¹)': 'pm_1_sca_450',
    'Aerosol light scattering coefficient (Mm⁻¹).1': 'pm_1_sca_550',
    'Aerosol light scattering coefficient (Mm⁻¹).2': 'pm_1_sca_700',
    'Aerosol light backwards-hemispheric scattering coefficient (Mm⁻¹)': 'pm_1_bsca_450',
    'Aerosol light backwards-hemispheric scattering coefficient (Mm⁻¹).1': 'pm_1_bsca_550',
    'Aerosol light backwards-hemispheric scattering coefficient (Mm⁻¹).2': 'pm_1_bsca_700',
    'Sample temperature (°C)': 'pm_1_sample_temperature',
    'Sample RH (%)': 'pm_1_relative_humidity',
    'Sample pressure (hPa)': 'pm_1_sample_pressure'
    }

pm_10_2018_columns = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Semicolon delimited list of cut sizes present': 'pm_cut_size_range',
    'total light scattering coefficient at 450 nm (Mm-1)': 'pm_10_sca_450',
    'total light scattering coefficient at 550 nm (Mm-1)': 'pm_10_sca_550',
    'total light scattering coefficient at 700 nm (Mm-1)': 'pm_10_sca_700',
    'backwards hemispheric light scattering coefficient at 450 nm (Mm-1)': 'pm_10_bsca_450',
    'backwards hemispheric light scattering coefficient at 550 nm (Mm-1)': 'pm_10_bsca_550',
    'backwards hemispheric light scattering coefficient at 700 nm (Mm-1)': 'pm_10_bsca_700',
    'measurement cell temperature (°C)': 'pm_10_sample_temperature',
    'measurement cell relative humidity (%)': 'pm_10_relative_humidity',
    'measurement cell pressure (hPa)': 'pm_10_sample_pressure'
    }

pm_10_2022_columns = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Semicolon delimited list of cut sizes present': 'pm_cut_size_range',
    'Aerosol light scattering coefficient (Mm⁻¹)': 'pm_10_sca_450',
    'Aerosol light scattering coefficient (Mm⁻¹).1': 'pm_10_sca_550',
    'Aerosol light scattering coefficient (Mm⁻¹).2': 'pm_10_sca_700',
    'Aerosol light backwards-hemispheric scattering coefficient (Mm⁻¹)': 'pm_10_bsca_450',
    'Aerosol light backwards-hemispheric scattering coefficient (Mm⁻¹).1': 'pm_10_bsca_550',
    'Aerosol light backwards-hemispheric scattering coefficient (Mm⁻¹).2': 'pm_10_bsca_700',
    'Sample temperature (°C)': 'pm_10_sample_temperature',
    'Sample RH (%)': 'pm_10_relative_humidity',
    'Sample pressure (hPa)': 'pm_10_sample_pressure'
    }

pm_1_sca_2018 = pm_1_sca_2018.rename(columns = pm_1_2018_columns)
pm_1_sca_2022 = pm_1_sca_2022.rename(columns = pm_1_2022_columns)

pm_10_sca_2018 = pm_10_sca_2018.rename(columns = pm_10_2018_columns)
pm_10_sca_2022 = pm_10_sca_2022.rename(columns = pm_10_2022_columns)

combined_data.iloc[:, 0] = pd.to_datetime(combined_data.iloc[:, 0])
combined_data.iloc[:, 1:] = combined_data.iloc[:, 1:].astype(float)
combined_data = combined_data.set_index('date_and_time')
combined_data = combined_data.resample('1H').mean()
combined_data.index = combined_data.index + pd.Timedelta(minutes=30)
combined_data = combined_data.reset_index()
combined_data['date_and_time'] = pd.to_datetime(combined_data['date_and_time'])
combined_data.iloc[:, 1:] = combined_data.iloc[:, 1:].astype(float)
combined_data = combined_data[combined_data['dmps_file_relative_humidity'] <= 40]

pm_1_sca_2018.iloc[:, 0] = pd.to_datetime(pm_1_sca_2018.iloc[:, 0])
pm_1_sca_2022.iloc[:, 0] = pd.to_datetime(pm_1_sca_2022.iloc[:, 0])
pm_1_sca_2018.iloc[:, 1:] = pm_1_sca_2018.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
pm_1_sca_2022.iloc[:, 1:] = pm_1_sca_2022.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
pm_1_sca_2018 = pm_1_sca_2018.iloc[:-1, :]
pm_1_sca_2018 = pm_1_sca_2018.set_index('date_and_time')    
pm_1_sca_2022 = pm_1_sca_2022.set_index('date_and_time')
pm_1_sca_2018 = pm_1_sca_2018.resample('1H').mean()
pm_1_sca_2018.index += pd.Timedelta(minutes=30)
pm_1_sca_2022 = pm_1_sca_2022.resample('1H').mean()
pm_1_sca_2022.index += pd.Timedelta(minutes=30)
pm_1_sca_2018 = pm_1_sca_2018.reset_index()
pm_1_sca_2022 = pm_1_sca_2022.reset_index()
pm_1_sca_2018['date_and_time'] = pd.to_datetime(pm_1_sca_2018['date_and_time'])
pm_1_sca_2018.iloc[:, 1:] = pm_1_sca_2018.iloc[:, 1:].astype(float)
pm_1_sca_2022['date_and_time'] = pd.to_datetime(pm_1_sca_2022['date_and_time'])
pm_1_sca_2022.iloc[:, 1:] = pm_1_sca_2022.iloc[:, 1:].astype(float)

pm_10_sca_2018.iloc[:, 0] = pd.to_datetime(pm_10_sca_2018.iloc[:, 0])
pm_10_sca_2022.iloc[:, 0] = pd.to_datetime(pm_10_sca_2022.iloc[:, 0])
pm_10_sca_2018.iloc[:, 1:] = pm_10_sca_2018.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
pm_10_sca_2022.iloc[:, 1:] = pm_10_sca_2022.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
pm_10_sca_2018 = pm_10_sca_2018.iloc[:-1, :]
pm_10_sca_2018 = pm_10_sca_2018.set_index('date_and_time')
pm_10_sca_2022 = pm_10_sca_2022.set_index('date_and_time')
pm_10_sca_2018 = pm_10_sca_2018.resample('30min', label='right', closed='right').mean()
pm_10_sca_2022 = pm_10_sca_2022.resample('30min', label='right', closed='right').mean()
pm_10_sca_2018 = pm_10_sca_2018.reset_index()
pm_10_sca_2022 = pm_10_sca_2022.reset_index()
pm_10_sca_2018['date_and_time'] = pd.to_datetime(pm_10_sca_2018['date_and_time'])
pm_10_sca_2018.iloc[:, 1:] = pm_10_sca_2018.iloc[:, 1:].astype(float)
pm_10_sca_2022['date_and_time'] = pd.to_datetime(pm_10_sca_2022['date_and_time'])
pm_10_sca_2022.iloc[:, 1:] = pm_10_sca_2022.iloc[:, 1:].astype(float)

pm_1_2018_2022 = pd.concat([pm_1_sca_2018, pm_1_sca_2022], axis=0).sort_values('date_and_time')

pm_10_2018_2022 = pd.concat([pm_10_sca_2018, pm_10_sca_2022], axis=0).sort_values('date_and_time')

pm_1_2018_2022 = pd.merge(
    pm_1_2018_2022,
    combined_data,
    on='date_and_time',
    how='outer'
)

pm_10_2018_2022 = pd.merge(
    pm_10_2018_2022,
    combined_data,
    on='date_and_time',
    how='outer'
)

cols_to_drop = [
    'pm_cut_size_range',
    'pm_1_sample_temperature', 'pm_1_relative_humidity',
    'pm_10_sample_temperature', 'pm_10_relative_humidity'
]

pm_1_2018_2022 = pm_1_2018_2022.drop(columns=[col for col in cols_to_drop if col in pm_1_2018_2022.columns])
pm_10_2018_2022 = pm_10_2018_2022.drop(columns=[col for col in cols_to_drop if col in pm_10_2018_2022.columns])

pm_1_2018_2022['pm_1_correction_factor'] = 0.2697/((pm_1_2018_2022['dmps_flw_file_temperature'] + 273.15)/pm_1_2018_2022['pm_1_sample_pressure'])

pm_10_2018_2022['pm_10_correction_factor'] = 0.2697/((pm_10_2018_2022['dmps_flw_file_temperature'] + 273.15)/pm_10_2018_2022['pm_10_sample_pressure'])

pm_1_sca_df = pd.read_csv('./data/sca/neph_sca_bsca_pm1_SMEARii_2010_2021.txt', sep=r'\s+')

pm_1_sca_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_1_sca_450', 'pm_1_sca_550', 'pm_1_sca_700', 'pm_1_bsca_450', 'pm_1_bsca_550', 'pm_1_bsca_700']
pm_1_sca_np = pm_1_sca_df.to_numpy()
pm_1_sca_df = pd.DataFrame(pm_1_sca_np)
pm_1_sca_df.columns = pm_1_sca_cols
pm_1_sca_datetime = pm_1_sca_df['year'].astype(int).astype(str) + '-' + pm_1_sca_df['month'].astype(int).astype(str) + '-' + pm_1_sca_df['date'].astype(int).astype(str) + ' ' + pm_1_sca_df['hour'].astype(int).astype(str) + ':' + pm_1_sca_df['minute'].astype(int).astype(str) + ':' + pm_1_sca_df['second'].astype(int).astype(str)
pm_1_sca_datetime = pm_1_sca_datetime.to_frame()
pm_1_sca_datetime.columns = ['date_and_time']
pm_1_sca_df = pm_1_sca_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_1_sca_frames = [pm_1_sca_datetime, pm_1_sca_df]
pm_1_sca_df = pd.concat(pm_1_sca_frames, axis = 1)
pm_1_sca_df['date_and_time'] = pd.to_datetime(pm_1_sca_df['date_and_time'])
pm_1_sca_df.iloc[:, 1:] = pm_1_sca_df.iloc[:, 1:].astype(float)

pm_1_sca_df = pm_1_sca_df[(pm_1_sca_df['date_and_time'] >= '2010-10-04 00:00:00') &
            (pm_1_sca_df['date_and_time'] <= '2018-01-01 00:00:00')]

pm_1_sca_df = pd.concat([pm_1_sca_df, pm_1_2018_2022])
pm_1_sca_df['date_and_time'] = pd.to_datetime(pm_1_sca_df['date_and_time'])
pm_1_sca_df.iloc[:, 1:] = pm_1_sca_df.iloc[:, 1:].astype(float)

date_and_time_range_pm1 = pm_1_sca_df['date_and_time'].dt.year <= 2017
pm_1_sca_df.loc[date_and_time_range_pm1, 'pm_1_correction_factor'] = np.where(
    pd.isna(pm_1_sca_df.loc[date_and_time_range_pm1, 'pm_1_correction_factor']),
    1,
    pm_1_sca_df.loc[date_and_time_range_pm1, 'pm_1_correction_factor']
)

pm_1_sca_df['pm_1_correction_factor'] = np.where(
    pd.isna(pm_1_sca_df['pm_1_correction_factor']),
    1,
    pm_1_sca_df['pm_1_correction_factor']
)

pm_1_sca_df.sort_values('date_and_time', inplace=True)

start_date = pd.to_datetime('2010-10-04 00:00:00')
end_date = pd.to_datetime('2022-10-04 00:00:00')

pm_1_sca_df = pm_1_sca_df[(pm_1_sca_df['date_and_time'] >= start_date) & 
                          (pm_1_sca_df['date_and_time'] <= end_date)]

pm_1_cols_to_correct = [
    'pm_1_sca_450', 'pm_1_sca_550', 'pm_1_sca_700',
    'pm_1_bsca_450', 'pm_1_bsca_550', 'pm_1_bsca_700'
]

pm_1_sca_df = pm_1_sca_df.dropna(
    subset=[
        'pm_1_sca_450', 'pm_1_sca_550', 'pm_1_sca_700',
        'pm_1_bsca_450', 'pm_1_bsca_550', 'pm_1_bsca_700'
    ]
)

pm_1_sca_df['pm_1_b_450'] = pm_1_sca_df['pm_1_bsca_450'] / pm_1_sca_df['pm_1_sca_450']
pm_1_sca_df['pm_1_b_550'] = pm_1_sca_df['pm_1_bsca_550'] / pm_1_sca_df['pm_1_sca_550']
pm_1_sca_df['pm_1_b_700'] = pm_1_sca_df['pm_1_bsca_700'] / pm_1_sca_df['pm_1_sca_700']

pm_10_sca_df = pd.read_csv('./data/sca/neph_sca_bsca_pm10_SMEARii_2006_2021.txt', sep=r'\s+')

pm_10_sca_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_10_sca_450', 'pm_10_sca_550', 'pm_10_sca_700', 'pm_10_bsca_450', 'pm_10_bsca_550', 'pm_10_bsca_700']
pm_10_sca_np = pm_10_sca_df.to_numpy()
pm_10_sca_df = pd.DataFrame(pm_10_sca_np)
pm_10_sca_df.columns = pm_10_sca_cols
pm_10_sca_datetime = pm_10_sca_df['year'].astype(int).astype(str) + '-' + pm_10_sca_df['month'].astype(int).astype(str) + '-' + pm_10_sca_df['date'].astype(int).astype(str) + ' ' + pm_10_sca_df['hour'].astype(int).astype(str) + ':' + pm_10_sca_df['minute'].astype(int).astype(str) + ':' + pm_10_sca_df['second'].astype(int).astype(str)
pm_10_sca_datetime = pm_10_sca_datetime.to_frame()
pm_10_sca_datetime.columns = ['date_and_time']
pm_10_sca_df = pm_10_sca_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_10_sca_frames = [pm_10_sca_datetime, pm_10_sca_df]
pm_10_sca_df = pd.concat(pm_10_sca_frames, axis = 1)
pm_10_sca_df['date_and_time'] = pd.to_datetime(pm_10_sca_df['date_and_time'])
pm_10_sca_df.iloc[:, 1:] = pm_10_sca_df.iloc[:, 1:].astype(float)

pm_10_sca_df = pd.concat([pm_10_sca_df, pm_10_2018_2022])
pm_10_sca_df['date_and_time'] = pd.to_datetime(pm_10_sca_df['date_and_time'])
pm_10_sca_df.iloc[:, 1:] = pm_10_sca_df.iloc[:, 1:].astype(float)

date_and_time_range_pm10 = pm_10_sca_df['date_and_time'].dt.year <= 2017
pm_10_sca_df.loc[date_and_time_range_pm10, 'pm_10_correction_factor'] = np.where(
    pd.isna(pm_10_sca_df.loc[date_and_time_range_pm10, 'pm_10_correction_factor']),
    1,
    pm_10_sca_df.loc[date_and_time_range_pm10, 'pm_10_correction_factor']
)

pm_10_sca_df['pm_10_correction_factor'] = np.where(
    pd.isna(pm_10_sca_df['pm_10_correction_factor']),
    1,
    pm_10_sca_df['pm_10_correction_factor']
)

pm_10_sca_df.sort_values('date_and_time', inplace=True)

start_date = pd.to_datetime('2010-10-04 00:00:00')
end_date = pd.to_datetime('2022-10-04 00:00:00')

pm_10_sca_df = pm_10_sca_df[(pm_10_sca_df['date_and_time'] >= start_date) & 
                          (pm_10_sca_df['date_and_time'] <= end_date)]

pm_10_cols_to_correct = [
    'pm_10_sca_450', 'pm_10_sca_550', 'pm_10_sca_700',
    'pm_10_bsca_450', 'pm_10_bsca_550', 'pm_10_bsca_700'
]

pm_10_sca_df[pm_10_cols_to_correct] = pm_10_sca_df[pm_10_cols_to_correct].multiply(
    pm_10_sca_df['pm_10_correction_factor'], axis=0
)

pm_10_sca_df = pm_10_sca_df.dropna(
    subset=[
        'pm_10_sca_450', 'pm_10_sca_550', 'pm_10_sca_700',
        'pm_10_bsca_450', 'pm_10_bsca_550', 'pm_10_bsca_700'
    ]
)

pm_10_sca_df['pm_10_b_450'] = pm_10_sca_df['pm_10_bsca_450'] / pm_10_sca_df['pm_10_sca_450']
pm_10_sca_df['pm_10_b_550'] = pm_10_sca_df['pm_10_bsca_550'] / pm_10_sca_df['pm_10_sca_550']
pm_10_sca_df['pm_10_b_700'] = pm_10_sca_df['pm_10_bsca_700'] / pm_10_sca_df['pm_10_sca_700']

sca_10_1_df = pd.merge(
    pm_1_sca_df,
    pm_10_sca_df,
    on='date_and_time',
    how='outer',
    suffixes=('_x', '_y')
)

for col in sca_10_1_df.columns:
    if col.endswith('_x'):
        base_col = col[:-2]
        col_y = f"{base_col}_y"

        if col_y in sca_10_1_df.columns:
            if sca_10_1_df[col].equals(sca_10_1_df[col_y]):
                sca_10_1_df.drop(columns=[col_y], inplace=True)
                sca_10_1_df.rename(columns={col: base_col}, inplace=True)

sca_10_1_df['date_and_time'] = pd.to_datetime(sca_10_1_df['date_and_time'])

sca_10_1_df.iloc[:, 1:] = sca_10_1_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

start_date = pd.to_datetime('2010-10-04 00:00:00')
end_date = pd.to_datetime('2022-10-04 00:00:00')

sca_10_1_df = sca_10_1_df[(sca_10_1_df['date_and_time'] >= start_date) & 
                           (sca_10_1_df['date_and_time'] <= end_date)]

sca_10_1_df['date_only'] = sca_10_1_df['date_and_time'].dt.floor('D')

valid_data_per_day = sca_10_1_df.groupby('date_only').count()['date_and_time']

number_of_valid_days = valid_data_per_day[valid_data_per_day >= 18].index

sca_10_1_df = sca_10_1_df[sca_10_1_df['date_only'].isin(number_of_valid_days)]

sca_10_1_df = sca_10_1_df.groupby('date_only').mean(numeric_only=True)

sca_10_1_df = sca_10_1_df.reset_index()

sca_10_1_df = sca_10_1_df.rename(columns={'date_only': 'date_and_time'})

sca_10_1_df['date_and_time'] = sca_10_1_df['date_and_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

pm_10_sae_df = sca_10_1_df[['date_and_time', 'pm_10_sca_450', 'pm_10_sca_550', 'pm_10_sca_700']].copy()
pm_10_sae_df['date_and_time'] = pd.to_datetime(pm_10_sae_df['date_and_time'], errors='coerce')

pm_1_sae_df = sca_10_1_df[['date_and_time', 'pm_1_sca_450', 'pm_1_sca_550', 'pm_1_sca_700']].copy()
pm_1_sae_df['date_and_time'] = pd.to_datetime(pm_1_sae_df['date_and_time'], errors='coerce')

sca_10_1_df['bsca_10_1_450'] = sca_10_1_df['pm_10_bsca_450'] - sca_10_1_df['pm_1_bsca_450']
sca_10_1_df['bsca_10_1_550'] = sca_10_1_df['pm_10_bsca_550'] - sca_10_1_df['pm_1_bsca_550']
sca_10_1_df['bsca_10_1_700'] = sca_10_1_df['pm_10_bsca_700'] - sca_10_1_df['pm_1_bsca_700']

sca_10_1_df['sca_10_1_450'] = sca_10_1_df['pm_10_sca_450'] - sca_10_1_df['pm_1_sca_450']
sca_10_1_df['sca_10_1_550'] = sca_10_1_df['pm_10_sca_550'] - sca_10_1_df['pm_1_sca_550']
sca_10_1_df['sca_10_1_700'] = sca_10_1_df['pm_10_sca_700'] - sca_10_1_df['pm_1_sca_700']

sae_10_1_df = sca_10_1_df[['date_and_time', 'sca_10_1_450', 'sca_10_1_550', 'sca_10_1_700']].copy()
sae_10_1_df['date_and_time'] = pd.to_datetime(sae_10_1_df['date_and_time'], errors='coerce')

pm_10_bsae_df = sca_10_1_df[['date_and_time', 'pm_10_bsca_450', 'pm_10_bsca_550', 'pm_10_bsca_700']].copy()
pm_10_bsae_df['date_and_time'] = pd.to_datetime(pm_10_bsae_df['date_and_time'], errors='coerce')

pm_1_bsae_df = sca_10_1_df[['date_and_time', 'pm_1_bsca_450', 'pm_1_bsca_550', 'pm_1_bsca_700']].copy()
pm_1_bsae_df['date_and_time'] = pd.to_datetime(pm_1_bsae_df['date_and_time'], errors='coerce')

bsae_10_1_df = sca_10_1_df[['date_and_time', 'bsca_10_1_450', 'bsca_10_1_550', 'bsca_10_1_700']].copy()
bsae_10_1_df['date_and_time'] = pd.to_datetime(bsae_10_1_df['date_and_time'], errors='coerce')

def custom_round_sae_1(value):
    value_str = f"{value:.3f}"
    last_digit = int(value_str[-1])

    if last_digit in [0, 1, 2, 3, 4]:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP) + Decimal("0.01")

    return float(rounded_value)

def generate_y_ticks_sae_1(y_limits):
    y_min, y_max = y_limits
    y_ticks_raw = np.linspace(y_min, y_max, num=6)
    y_ticks = [custom_round_sae_1(tick) for tick in y_ticks_raw]
    return y_ticks

def format_number_sae_1(value, decimals=2):
    if abs(value) >= 100 or (abs(value) < 0.01 and value != 0):
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10**exponent)
        return f"{coefficient:.{decimals}f} × 10$^{{{exponent}}}$"
    else:
        quantization_format = f"1.{'0' * decimals}"
        return f"{Decimal(str(value)).quantize(Decimal(quantization_format), rounding=ROUND_HALF_UP):.{decimals}f}"

def format_yaxis_sae_1(value, _):
    return f"{value:.2f}"

def compute_sae_1(df, name):

    if 'date_and_time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'date_and_time' column.")

    numeric_cols = [col for col in df.columns if col != 'date_and_time']

    df_numeric = df[numeric_cols].copy().where(df[numeric_cols] > 0)

    wavelengths = np.array([450, 550, 700])
    log_lambda = np.log(wavelengths).reshape(-1, 1)

    valid_rows = df_numeric.dropna().astype(float)

    log_sigma = np.log(valid_rows)

    if log_sigma.shape[0] < 2:
        raise ValueError("Not enough valid rows for regression.")

    X = sm.add_constant(log_lambda)
    Y = log_sigma.values.T

    model = sm.OLS(Y, X).fit()
    slopes = model.params[1, :]

    sae_df = df[['date_and_time']].copy()
    sae_df[name] = np.nan
    sae_df.loc[valid_rows.index, name] = -slopes

    return sae_df

sae_0 = compute_sae_1(pm_10_sae_df, 'sae_0')
sae_1 = compute_sae_1(pm_1_sae_df, 'sae_1') 
sae_2 = compute_sae_1(sae_10_1_df, 'sae_2')

def custom_round_bsae_1(value):
    value_str = f"{value:.3f}"
    last_digit = int(value_str[-1])

    if last_digit in [0, 1, 2, 3, 4]:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP) + Decimal("0.01")

    return float(rounded_value)

def generate_y_ticks_bsae_1(y_limits):
    y_min, y_max = y_limits
    y_ticks_raw = np.linspace(y_min, y_max, num=6)
    y_ticks = [custom_round_sae_1(tick) for tick in y_ticks_raw]
    return y_ticks

def format_number_bsae_1(value, decimals=2):
    if abs(value) >= 100 or (abs(value) < 0.01 and value != 0):
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10**exponent)
        return f"{coefficient:.{decimals}f} × 10$^{{{exponent}}}$"
    else:
        quantization_format = f"1.{'0' * decimals}"
        return f"{Decimal(str(value)).quantize(Decimal(quantization_format), rounding=ROUND_HALF_UP):.{decimals}f}"

def format_yaxis_bsae_1(value, _):
    return f"{value:.2f}"

def compute_bsae_1(df, name):
    if 'date_and_time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'date_and_time' column.")
        
    numeric_cols = [col for col in df.columns if col != 'date_and_time']

    df_numeric = df[numeric_cols].copy().where(df[numeric_cols] > 0)

    wavelengths = np.array([450, 550, 700])
    log_lambda = np.log(wavelengths).reshape(-1, 1)

    valid_rows = df_numeric.dropna().astype(float)

    log_sigma = np.log(valid_rows)

    if log_sigma.shape[0] < 2:
        raise ValueError("Not enough valid rows for regression.")

    X = sm.add_constant(log_lambda)
    Y = log_sigma.values.T

    model = sm.OLS(Y, X).fit()
    slopes = model.params[1, :]

    bsae_df = df[['date_and_time']].copy()
    bsae_df[name] = np.nan
    bsae_df.loc[valid_rows.index, name] = -slopes

    return bsae_df

bsae_0 = compute_bsae_1(pm_10_bsae_df, 'bsae_0')
bsae_1 = compute_bsae_1(pm_1_bsae_df, 'bsae_1')
bsae_2 = compute_bsae_1(bsae_10_1_df, 'bsae_2')

sca_10_1_df['date_and_time'] = pd.to_datetime(sca_10_1_df['date_and_time'], errors='coerce')
sca_10_1_df = sca_10_1_df.sort_values(by = 'date_and_time')
sca_10_1_df.iloc[:, 1:] = sca_10_1_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

sae_0['date_and_time'] = pd.to_datetime(sae_0['date_and_time'], errors='coerce')
sae_0.iloc[:, 1:] = sae_0.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
sae_1['date_and_time'] = pd.to_datetime(sae_1['date_and_time'], errors='coerce')
sae_1.iloc[:, 1:] = sae_1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
sae_2['date_and_time'] = pd.to_datetime(sae_2['date_and_time'], errors='coerce')
sae_2.iloc[:, 1:] = sae_2.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

bsae_0['date_and_time'] = pd.to_datetime(bsae_0['date_and_time'], errors='coerce')
bsae_0.iloc[:, 1:] = bsae_0.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
bsae_1['date_and_time'] = pd.to_datetime(bsae_1['date_and_time'], errors='coerce')
bsae_1.iloc[:, 1:] = bsae_1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
bsae_2['date_and_time'] = pd.to_datetime(bsae_2['date_and_time'], errors='coerce')
bsae_2.iloc[:, 1:] = bsae_2.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

sca_10_1_df['date_and_time'] = pd.to_datetime(sca_10_1_df['date_and_time'], errors='coerce')
sae_0['date_and_time'] = pd.to_datetime(sae_0['date_and_time'], errors='coerce')
sae_1['date_and_time'] = pd.to_datetime(sae_1['date_and_time'], errors='coerce')
sae_2['date_and_time'] = pd.to_datetime(sae_2['date_and_time'], errors='coerce')
bsae_0['date_and_time'] = pd.to_datetime(bsae_0['date_and_time'], errors='coerce')
bsae_1['date_and_time'] = pd.to_datetime(bsae_1['date_and_time'], errors='coerce')
bsae_2['date_and_time'] = pd.to_datetime(bsae_2['date_and_time'], errors='coerce')

sca_10_1_df.sort_values('date_and_time', inplace=True)
sae_0.sort_values('date_and_time', inplace=True)
sae_1.sort_values('date_and_time', inplace=True)
sae_2.sort_values('date_and_time', inplace=True)
bsae_0.sort_values('date_and_time', inplace=True)
bsae_1.sort_values('date_and_time', inplace=True)
bsae_2.sort_values('date_and_time', inplace=True)

sca_10_1_df = pd.merge_asof(sca_10_1_df, sae_0, on='date_and_time')
sca_10_1_df = pd.merge_asof(sca_10_1_df, sae_1, on='date_and_time')
sca_10_1_df = pd.merge_asof(sca_10_1_df, sae_2, on='date_and_time')
sca_10_1_df = pd.merge_asof(sca_10_1_df, bsae_0, on='date_and_time')
sca_10_1_df = pd.merge_asof(sca_10_1_df, bsae_1, on='date_and_time')
sca_10_1_df = pd.merge_asof(sca_10_1_df, bsae_2, on='date_and_time')

sca_10_1_df['date_and_time'] = pd.to_datetime(sca_10_1_df['date_and_time'], errors='coerce')
sca_10_1_df = sca_10_1_df.sort_values(by = 'date_and_time')
sca_10_1_df.iloc[:, 1:] = sca_10_1_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
sca_10_1_df.set_index('date_and_time', inplace=True)

def truncation_corr(df, wavelengths, pm_prefix):
    sae_column = "sae_0" if "pm_10" in pm_prefix else "sae_1"

    for wl in wavelengths:
        correction_col = f"{pm_prefix}_{wl}_correction_factor"
        corrected_col = f"{pm_prefix}_{wl}_truncation_corrected"

        df[correction_col] = np.nan

        mask_2010_2017 = (df.index.year >= 2010) & (df.index.year <= 2017) & df[f"{pm_prefix}_{wl}"].notna()
        df.loc[mask_2010_2017, correction_col] = 1

        mask_2018_2022 = (df.index.year >= 2018) & (df.index.year <= 2022) & df[f"{pm_prefix}_{wl}"].notna()

        if "pm_10" in pm_prefix:
            correction_factors = {
                450: 1.365 - 0.156 * df.loc[mask_2018_2022, sae_column],
                550: 1.337 - 0.138 * df.loc[mask_2018_2022, sae_column],
                700: 1.297 - 0.113 * df.loc[mask_2018_2022, sae_column],
            }
        else:
            correction_factors = {
                450: 1.165 - 0.046 * df.loc[mask_2018_2022, sae_column],
                550: 1.152 - 0.044 * df.loc[mask_2018_2022, sae_column],
                700: 1.120 - 0.035 * df.loc[mask_2018_2022, sae_column],
            }

        df.loc[mask_2018_2022, correction_col] = np.maximum(correction_factors[wl], 1)

        df.loc[df[f"{pm_prefix}_{wl}"].notna(), corrected_col] = df.loc[df[f"{pm_prefix}_{wl}"].notna(), f"{pm_prefix}_{wl}"] * df.loc[df[f"{pm_prefix}_{wl}"].notna(), correction_col]

    return df

wavelengths = [450, 550, 700]
sca_10_1_df = truncation_corr(sca_10_1_df, wavelengths, "pm_10_sca")
sca_10_1_df = truncation_corr(sca_10_1_df, wavelengths, "pm_1_sca")

sca_10_1_df.reset_index(inplace=True)

columns_to_keep = [
    'date_and_time','pm_10_sca_450', 'pm_10_sca_550', 
    'pm_10_sca_700', 'pm_1_sca_450', 'pm_1_sca_550', 
    'pm_1_sca_700', 'pm_10_sca_450_truncation_corrected', 'pm_10_sca_550_truncation_corrected',
    'pm_10_sca_700_truncation_corrected', 'pm_10_sca_450_correction_factor',
    'pm_10_sca_550_correction_factor', 'pm_10_sca_700_correction_factor',
    'pm_1_sca_450_truncation_corrected', 'pm_1_sca_550_truncation_corrected',
    'pm_1_sca_700_truncation_corrected', 'pm_1_sca_450_correction_factor',
    'pm_1_sca_550_correction_factor', 'pm_1_sca_700_correction_factor',
    'pm_10_bsca_450', 'pm_10_bsca_550', 'pm_10_bsca_700',
    'pm_1_bsca_450', 'pm_1_bsca_550', 'pm_1_bsca_700'
]

sca_10_1_df = sca_10_1_df[columns_to_keep]

def plot_correction_factors(df, pm_prefix, wavelengths):
    for wl in wavelengths:
        correction_col = f"{pm_prefix}_{wl}_correction_factor"
        original_col = f"{pm_prefix}_{wl}"
        corrected_col = f"{pm_prefix}_{wl}_truncation_corrected"

        plt.figure(figsize=(15, 5))
        plt.plot(df['date_and_time'], df[correction_col], 'o', markersize=3, alpha=0.5, label=f'Correction Factor {wl} nm', color='green')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Correction Factor', fontsize=14)
        plt.title(f'Truncation Correction Factor for {pm_prefix} at {wl} nm (2010-2022)', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(df['date_and_time'], df[original_col], 'o', markersize=3, alpha=0.5, label=f'Original {pm_prefix} {wl} nm', color='red')
        plt.plot(df['date_and_time'], df[corrected_col], 'o', markersize=3, alpha=0.5, label=f'Corrected {pm_prefix} {wl} nm', color='cornflowerblue')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Scattering Coefficient (Mm⁻¹)', fontsize=14)
        plt.title(f'Original vs Corrected Scattering for {pm_prefix} at {wl} nm (2010-2022)', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

plot_correction_factors(sca_10_1_df, "pm_10_sca", wavelengths)
plot_correction_factors(sca_10_1_df, "pm_1_sca", wavelengths)

sca_10_1_df.drop(columns=[
    'pm_10_sca_450', 'pm_10_sca_550', 'pm_10_sca_700',
    'pm_1_sca_450', 'pm_1_sca_550', 'pm_1_sca_700',
    'pm_10_sca_450_correction_factor', 'pm_10_sca_550_correction_factor', 'pm_10_sca_700_correction_factor',
    'pm_1_sca_450_correction_factor', 'pm_1_sca_550_correction_factor', 'pm_1_sca_700_correction_factor'
], inplace=True)

sca_10_1_df.rename(columns=lambda x: x.replace('_truncation_corrected', ''), inplace=True)

pm_10_sae_df = sca_10_1_df[['date_and_time', 'pm_10_sca_450', 'pm_10_sca_550', 'pm_10_sca_700']].copy()
pm_10_sae_df['date_and_time'] = pd.to_datetime(pm_10_sae_df['date_and_time'], errors='coerce')

pm_1_sae_df = sca_10_1_df[['date_and_time', 'pm_1_sca_450', 'pm_1_sca_550', 'pm_1_sca_700']].copy()
pm_1_sae_df['date_and_time'] = pd.to_datetime(pm_1_sae_df['date_and_time'], errors='coerce')

sca_10_1_df['bsca_10_1_450'] = sca_10_1_df['pm_10_bsca_450'] - sca_10_1_df['pm_1_bsca_450']
sca_10_1_df['bsca_10_1_550'] = sca_10_1_df['pm_10_bsca_550'] - sca_10_1_df['pm_1_bsca_550']
sca_10_1_df['bsca_10_1_700'] = sca_10_1_df['pm_10_bsca_700'] - sca_10_1_df['pm_1_bsca_700']

sca_10_1_df['sca_10_1_450'] = sca_10_1_df['pm_10_sca_450'] - sca_10_1_df['pm_1_sca_450']
sca_10_1_df['sca_10_1_550'] = sca_10_1_df['pm_10_sca_550'] - sca_10_1_df['pm_1_sca_550']
sca_10_1_df['sca_10_1_700'] = sca_10_1_df['pm_10_sca_700'] - sca_10_1_df['pm_1_sca_700']

sae_10_1_df = sca_10_1_df[['date_and_time', 'sca_10_1_450', 'sca_10_1_550', 'sca_10_1_700']].copy()
sae_10_1_df['date_and_time'] = pd.to_datetime(sae_10_1_df['date_and_time'], errors='coerce')

pm_10_bsae_df = sca_10_1_df[['date_and_time', 'pm_10_bsca_450', 'pm_10_bsca_550', 'pm_10_bsca_700']].copy()
pm_10_bsae_df['date_and_time'] = pd.to_datetime(pm_10_bsae_df['date_and_time'], errors='coerce')

pm_1_bsae_df = sca_10_1_df[['date_and_time', 'pm_1_bsca_450', 'pm_1_bsca_550', 'pm_1_bsca_700']].copy()
pm_1_bsae_df['date_and_time'] = pd.to_datetime(pm_1_bsae_df['date_and_time'], errors='coerce')

bsae_10_1_df = sca_10_1_df[['date_and_time', 'bsca_10_1_450', 'bsca_10_1_550', 'bsca_10_1_700']].copy()
bsae_10_1_df['date_and_time'] = pd.to_datetime(bsae_10_1_df['date_and_time'], errors='coerce')

sca_10_1_df['date_and_time'] = pd.to_datetime(sca_10_1_df['date_and_time'], errors='coerce')
sca_10_1_df = sca_10_1_df.sort_values(by = 'date_and_time')

sca_10_1_df.iloc[:, 1:] = sca_10_1_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

def custom_round_sae_2(value):
    value_str = f"{value:.3f}"
    last_digit = int(value_str[-1])

    if last_digit in [0, 1, 2, 3, 4]:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP) + Decimal("0.01")

    return float(rounded_value)

def generate_y_ticks_sae_2(y_limits):
    y_min, y_max = y_limits
    y_ticks_raw = np.linspace(y_min, y_max, num=6)
    y_ticks = [custom_round_sae_1(tick) for tick in y_ticks_raw]
    return y_ticks

def format_number_sae_2(value, decimals=2):
    if abs(value) >= 100 or (abs(value) < 0.01 and value != 0):
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10**exponent)
        return f"{coefficient:.{decimals}f} × 10$^{{{exponent}}}$"
    else:
        quantization_format = f"1.{'0' * decimals}"
        return f"{Decimal(str(value)).quantize(Decimal(quantization_format), rounding=ROUND_HALF_UP):.{decimals}f}"

def format_yaxis_sae_2(value, _):
    return f"{value:.2f}"

def compute_sae_2(df, name):

    if 'date_and_time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'date_and_time' column.")

    numeric_cols = [col for col in df.columns if col != 'date_and_time']

    df_numeric = df[numeric_cols].copy().where(df[numeric_cols] > 0)

    wavelengths = np.array([450, 550, 700])
    log_lambda = np.log(wavelengths).reshape(-1, 1)

    valid_rows = df_numeric.dropna().astype(float)

    log_sigma = np.log(valid_rows)

    if log_sigma.shape[0] < 2:
        raise ValueError("Not enough valid rows for regression.")

    X = sm.add_constant(log_lambda)
    Y = log_sigma.values.T

    model = sm.OLS(Y, X).fit()
    slopes = model.params[1, :]

    sae_df = df[['date_and_time']].copy()
    sae_df[name] = np.nan
    sae_df.loc[valid_rows.index, name] = -slopes

    return sae_df

bsae_0 = compute_bsae_1(pm_10_bsae_df, 'bsae_0')
bsae_1 = compute_bsae_1(pm_1_bsae_df, 'bsae_1')
bsae_2 = compute_bsae_1(bsae_10_1_df, 'bsae_2')

sae_0 = compute_sae_2(pm_10_sae_df, 'sae_0')
sae_1 = compute_sae_2(pm_1_sae_df, 'sae_1') 
sae_2 = compute_sae_2(sae_10_1_df, 'sae_2')

bsae_0['date_and_time'] = pd.to_datetime(bsae_0['date_and_time'], errors='coerce')
bsae_0.iloc[:, 1:] = bsae_0.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
bsae_1['date_and_time'] = pd.to_datetime(bsae_1['date_and_time'], errors='coerce')
bsae_1.iloc[:, 1:] = bsae_1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
bsae_2['date_and_time'] = pd.to_datetime(bsae_2['date_and_time'], errors='coerce')
bsae_2.iloc[:, 1:] = bsae_2.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

sae_0['date_and_time'] = pd.to_datetime(sae_0['date_and_time'], errors='coerce')
sae_0.iloc[:, 1:] = sae_0.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
sae_1['date_and_time'] = pd.to_datetime(sae_1['date_and_time'], errors='coerce')
sae_1.iloc[:, 1:] = sae_1.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
sae_2['date_and_time'] = pd.to_datetime(sae_2['date_and_time'], errors='coerce')
sae_2.iloc[:, 1:] = sae_2.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

sae_0['date_and_time'] = pd.to_datetime(sae_0['date_and_time'], errors='coerce')
sae_1['date_and_time'] = pd.to_datetime(sae_1['date_and_time'], errors='coerce')
sae_2['date_and_time'] = pd.to_datetime(sae_2['date_and_time'], errors='coerce')
bsae_0['date_and_time'] = pd.to_datetime(bsae_0['date_and_time'], errors='coerce')
bsae_1['date_and_time'] = pd.to_datetime(bsae_1['date_and_time'], errors='coerce')
bsae_2['date_and_time'] = pd.to_datetime(bsae_2['date_and_time'], errors='coerce')

sca_10_1_df.sort_values('date_and_time', inplace=True)
sae_0.sort_values('date_and_time', inplace=True)
sae_1.sort_values('date_and_time', inplace=True)
sae_2.sort_values('date_and_time', inplace=True)
bsae_0.sort_values('date_and_time', inplace=True)
bsae_1.sort_values('date_and_time', inplace=True)
bsae_2.sort_values('date_and_time', inplace=True)

sca_10_1_df = pd.merge(sca_10_1_df, sae_0, on='date_and_time', how='left')
sca_10_1_df = pd.merge(sca_10_1_df, sae_1, on='date_and_time', how='left')
sca_10_1_df = pd.merge(sca_10_1_df, sae_2, on='date_and_time', how='left')
sca_10_1_df = pd.merge(sca_10_1_df, bsae_0, on='date_and_time', how='left')
sca_10_1_df = pd.merge(sca_10_1_df, bsae_1, on='date_and_time', how='left')
sca_10_1_df = pd.merge(sca_10_1_df, bsae_2, on='date_and_time', how='left')

sca_10_1_df['pm_10_b_450'] = sca_10_1_df['pm_10_bsca_450'] / sca_10_1_df['pm_10_sca_450']
sca_10_1_df['pm_10_b_550'] = sca_10_1_df['pm_10_bsca_550'] / sca_10_1_df['pm_10_sca_550']
sca_10_1_df['pm_10_b_700'] = sca_10_1_df['pm_10_bsca_700'] / sca_10_1_df['pm_10_sca_700']

sca_10_1_df['pm_1_b_450'] = sca_10_1_df['pm_1_bsca_450'] / sca_10_1_df['pm_1_sca_450']
sca_10_1_df['pm_1_b_550'] = sca_10_1_df['pm_1_bsca_550'] / sca_10_1_df['pm_1_sca_550']
sca_10_1_df['pm_1_b_700'] = sca_10_1_df['pm_1_bsca_700'] / sca_10_1_df['pm_1_sca_700']

sca_10_1_df['b_10_1_450'] = (sca_10_1_df['pm_10_bsca_450'] - sca_10_1_df['pm_1_bsca_450']) / (sca_10_1_df['pm_10_sca_450'] - sca_10_1_df['pm_1_sca_450'])
sca_10_1_df['b_10_1_550'] = (sca_10_1_df['pm_10_bsca_550'] - sca_10_1_df['pm_1_bsca_550']) / (sca_10_1_df['pm_10_sca_550'] - sca_10_1_df['pm_1_sca_550'])
sca_10_1_df['b_10_1_700'] = (sca_10_1_df['pm_10_bsca_700'] - sca_10_1_df['pm_1_bsca_700']) / (sca_10_1_df['pm_10_sca_700'] - sca_10_1_df['pm_1_sca_700'])

sca_10_1_df['bsca_10_1_450'] = (sca_10_1_df['pm_10_bsca_450'] - sca_10_1_df['pm_1_bsca_450'])
sca_10_1_df['bsca_10_1_550'] = (sca_10_1_df['pm_10_bsca_550'] - sca_10_1_df['pm_1_bsca_550'])
sca_10_1_df['bsca_10_1_700'] = (sca_10_1_df['pm_10_bsca_700'] - sca_10_1_df['pm_1_bsca_700'])

sca_10_1_df['sca_10_1_450'] = (sca_10_1_df['pm_10_sca_450'] - sca_10_1_df['pm_1_sca_450'])
sca_10_1_df['sca_10_1_550'] = (sca_10_1_df['pm_10_sca_550'] - sca_10_1_df['pm_1_sca_550'])
sca_10_1_df['sca_10_1_700'] = (sca_10_1_df['pm_10_sca_700'] - sca_10_1_df['pm_1_sca_700'])

sca_10_1_df['b_10_1_450'] = (sca_10_1_df['pm_10_bsca_450'] - sca_10_1_df['pm_1_bsca_450']) / (sca_10_1_df['pm_10_sca_450'] - sca_10_1_df['pm_1_sca_450'])
sca_10_1_df['b_10_1_550'] = (sca_10_1_df['pm_10_bsca_550'] - sca_10_1_df['pm_1_bsca_550']) / (sca_10_1_df['pm_10_sca_550'] - sca_10_1_df['pm_1_sca_550'])
sca_10_1_df['b_10_1_700'] = (sca_10_1_df['pm_10_bsca_700'] - sca_10_1_df['pm_1_bsca_700']) / (sca_10_1_df['pm_10_sca_700'] - sca_10_1_df['pm_1_sca_700'])

pm_1_abs_df  = pd.read_csv('./data/ae31_abs_bc_pm1_SMEARii_2010_2017.txt', sep=r'\s+', header=None)
pm_1_abs_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950', 'pm_1_eBC_370', 'pm_1_eBC_470', 'pm_1_eBC_520', 'pm_1_eBC_590', 'pm_1_eBC_660', 'pm_1_eBC_880', 'pm_1_eBC_950']
pm_1_abs_np = pm_1_abs_df.to_numpy()
pm_1_abs_df = pd.DataFrame(pm_1_abs_np)
pm_1_abs_df.columns = pm_1_abs_cols
pm_1_abs_datetime = pm_1_abs_df['year'].astype(int).astype(str) + '-' + pm_1_abs_df['month'].astype(int).astype(str) + '-' + pm_1_abs_df['date'].astype(int).astype(str) + ' ' + pm_1_abs_df['hour'].astype(int).astype(str) + ':' + pm_1_abs_df['minute'].astype(int).astype(str) + ':' + pm_1_abs_df['second'].astype(int).astype(str)
pm_1_abs_datetime = pm_1_abs_datetime.to_frame()
pm_1_abs_datetime.columns = ['date_and_time']
pm_1_abs_df = pm_1_abs_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_1_abs_frames = [pm_1_abs_datetime, pm_1_abs_df]
pm_1_abs_df = pd.concat(pm_1_abs_frames, axis = 1)
pm_1_abs_df['date_and_time'] = pd.to_datetime(pm_1_abs_df['date_and_time'])
pm_1_abs_df.iloc[:, 1:] = pm_1_abs_df.iloc[:, 1:].astype(float)
pm_1_abs_df = pm_1_abs_df.iloc[:, :8]

pm_10_abs_df = pd.read_csv('./data/ae31_abs_bc_pm10_SMEARii_2006_2017.txt', sep=r'\s+')
pm_10_abs_cols = ['year', 'month', 'date', 'hour', 'minute', 'second', 'pm_10_abs_370', 'pm_10_abs_470', 'pm_10_abs_520', 'pm_10_abs_590', 'pm_10_abs_660', 'pm_10_abs_880', 'pm_10_abs_950', 'pm_10_eBC_370', 'pm_10_eBC_470', 'pm_10_eBC_520', 'pm_10_eBC_590', 'pm_10_eBC_660', 'pm_10_eBC_880', 'pm_10_eBC_950']
pm_10_abs_np = pm_10_abs_df.to_numpy()
pm_10_abs_df = pd.DataFrame(pm_10_abs_np)
pm_10_abs_df.columns = pm_10_abs_cols
pm_10_abs_datetime = pm_10_abs_df['year'].astype(int).astype(str) + '-' + pm_10_abs_df['month'].astype(int).astype(str) + '-' + pm_10_abs_df['date'].astype(int).astype(str) + ' ' + pm_10_abs_df['hour'].astype(int).astype(str) + ':' + pm_10_abs_df['minute'].astype(int).astype(str) + ':' + pm_10_abs_df['second'].astype(int).astype(str)
pm_10_abs_datetime = pm_10_abs_datetime.to_frame()
pm_10_abs_datetime.columns = ['date_and_time']
pm_10_abs_df = pm_10_abs_df.drop(columns = ['year', 'month', 'date', 'hour', 'minute', 'second'])
pm_10_abs_frames = [pm_10_abs_datetime, pm_10_abs_df]
pm_10_abs_df = pd.concat(pm_10_abs_frames, axis = 1)
pm_10_abs_df['date_and_time'] = pd.to_datetime(pm_10_abs_df['date_and_time'])
pm_10_abs_df.iloc[:, 1:] = pm_10_abs_df.iloc[:, 1:].astype(float)
pm_10_abs_df = pm_10_abs_df.iloc[:, :8]

ae_10_1_2018 = pd.read_csv(r'./data\abs\ae33\smr_20180101.csv')
ae_10_1_2019 = pd.read_csv(r'./data\abs\ae33\smr_20190101.csv')
ae_10_1_2020 = pd.read_csv(r'./data\abs\ae33\smr_20200101.csv')
ae_10_1_2021 = pd.read_csv(r'./data\abs\ae33\smr_20210101.csv')
ae_10_1_2022 = pd.read_csv(r'./data\abs\ae33\smr_20220101.csv')

def extract_abs_data(df_year):
    df_year = df_year.iloc[1:]

    df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'] = pd.to_datetime(df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'], format = '%Y-%m-%d %H:%M:%S')

    pm_1_range = [(3, 9), (23, 29), (43, 49)]
    pm_10_range = [(13, 19), (33, 39), (53, 59)]

    pm_1_data = df_year[df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'].dt.minute.isin([minute for start, end in pm_1_range for minute in range(start, end + 1)])]

    pm_10_data = df_year[df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'].dt.minute.isin([minute for start, end in pm_10_range for minute in range(start, end + 1)])]

    return pm_1_data, pm_10_data

pm_1_data_2018, pm_10_data_2018 = extract_abs_data(ae_10_1_2018)
pm_1_data_2019, pm_10_data_2019 = extract_abs_data(ae_10_1_2019)
pm_1_data_2020, pm_10_data_2020 = extract_abs_data(ae_10_1_2020)
pm_1_data_2021, pm_10_data_2021 = extract_abs_data(ae_10_1_2021)
pm_1_data_2022, pm_10_data_2022 = extract_abs_data(ae_10_1_2022)

pm_10_data_2018 = pm_10_data_2018.iloc[:, :8]
pm_10_data_2019 = pm_10_data_2019.iloc[:, :8]
pm_10_data_2020 = pm_10_data_2020.iloc[:, :8]
pm_10_data_2021 = pm_10_data_2021.iloc[:, :8]
pm_10_data_2022 = pm_10_data_2022.iloc[:, :8]

pm_1_data_2018 = pm_1_data_2018.iloc[:, :8]
pm_1_data_2019 = pm_1_data_2019.iloc[:, :8]
pm_1_data_2020 = pm_1_data_2020.iloc[:, :8]
pm_1_data_2021 = pm_1_data_2021.iloc[:, :8]
pm_1_data_2022 = pm_1_data_2022.iloc[:, :8]

pm_10_column_names = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹)': 'pm_10_abs_370',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).1': 'pm_10_abs_470',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).2': 'pm_10_abs_520',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).3': 'pm_10_abs_590',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).4': 'pm_10_abs_660',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).5': 'pm_10_abs_880',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).6': 'pm_10_abs_950'    
}

pm_1_column_names = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹)': 'pm_1_abs_370',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).1': 'pm_1_abs_470',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).2': 'pm_1_abs_520',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).3': 'pm_1_abs_590',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).4': 'pm_1_abs_660',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).5': 'pm_1_abs_880',
    'Aerosol light absorption coefficient on spot one (Mm⁻¹).6': 'pm_1_abs_950'
}

pm_1_data_2018 = pm_1_data_2018.rename(columns = pm_1_column_names)
pm_1_data_2019 = pm_1_data_2019.rename(columns = pm_1_column_names)
pm_1_data_2020 = pm_1_data_2020.rename(columns = pm_1_column_names)
pm_1_data_2021 = pm_1_data_2021.rename(columns = pm_1_column_names)
pm_1_data_2022 = pm_1_data_2022.rename(columns = pm_1_column_names)

pm_1_data_2018.iloc[:, 0] = pd.to_datetime(pm_1_data_2018.iloc[:, 0])
pm_1_data_2019.iloc[:, 0] = pd.to_datetime(pm_1_data_2019.iloc[:, 0])
pm_1_data_2020.iloc[:, 0] = pd.to_datetime(pm_1_data_2020.iloc[:, 0])
pm_1_data_2021.iloc[:, 0] = pd.to_datetime(pm_1_data_2021.iloc[:, 0])
pm_1_data_2022.iloc[:, 0] = pd.to_datetime(pm_1_data_2022.iloc[:, 0])

pm_1_data_2018.iloc[:, 1:] = pm_1_data_2018.iloc[:, 1:].astype(float)
pm_1_data_2019.iloc[:, 1:] = pm_1_data_2019.iloc[:, 1:].astype(float)
pm_1_data_2020.iloc[:, 1:] = pm_1_data_2020.iloc[:, 1:].astype(float)
pm_1_data_2021.iloc[:, 1:] = pm_1_data_2021.iloc[:, 1:].astype(float)
pm_1_data_2022.iloc[:, 1:] = pm_1_data_2022.iloc[:, 1:].astype(float)

pm_1_data_2018 = pm_1_data_2018.set_index('date_and_time')
pm_1_data_2019 = pm_1_data_2019.set_index('date_and_time')
pm_1_data_2020 = pm_1_data_2020.set_index('date_and_time')
pm_1_data_2021 = pm_1_data_2021.set_index('date_and_time')
pm_1_data_2022 = pm_1_data_2022.set_index('date_and_time')

pm_1_data_2018 = pm_1_data_2018.resample('H', offset = '30min').mean()
pm_1_data_2019 = pm_1_data_2019.resample('H', offset = '30min').mean()
pm_1_data_2020 = pm_1_data_2020.resample('H', offset = '30min').mean()
pm_1_data_2021 = pm_1_data_2021.resample('H', offset = '30min').mean()
pm_1_data_2022 = pm_1_data_2022.resample('H', offset = '30min').mean()

pm_1_data_2018 = pm_1_data_2018.reset_index()
pm_1_data_2019 = pm_1_data_2019.reset_index()
pm_1_data_2020 = pm_1_data_2020.reset_index()
pm_1_data_2021 = pm_1_data_2021.reset_index()
pm_1_data_2022 = pm_1_data_2022.reset_index()

pm_1_data_2018['date_and_time'] = pd.to_datetime(pm_1_data_2018['date_and_time'])
pm_1_data_2018.iloc[:, 1:] = pm_1_data_2018.iloc[:, 1:].astype(float)
pm_1_data_2019['date_and_time'] = pd.to_datetime(pm_1_data_2019['date_and_time'])
pm_1_data_2019.iloc[:, 1:] = pm_1_data_2019.iloc[:, 1:].astype(float)
pm_1_data_2020['date_and_time'] = pd.to_datetime(pm_1_data_2020['date_and_time'])
pm_1_data_2020.iloc[:, 1:] = pm_1_data_2020.iloc[:, 1:].astype(float)
pm_1_data_2021['date_and_time'] = pd.to_datetime(pm_1_data_2021['date_and_time'])
pm_1_data_2021.iloc[:, 1:] = pm_1_data_2021.iloc[:, 1:].astype(float)
pm_1_data_2022['date_and_time'] = pd.to_datetime(pm_1_data_2022['date_and_time'])
pm_1_data_2022.iloc[:, 1:] = pm_1_data_2022.iloc[:, 1:].astype(float)

pm_1_data = pd.concat([pm_1_data_2018, pm_1_data_2019, pm_1_data_2020, pm_1_data_2021, pm_1_data_2022])
pm_1_data = pm_1_data.iloc[:, :8]

pm_1_abs_list = []
pm_1_abs_list.append(pm_1_abs_df)
pm_1_abs_list.append(pm_1_data)
pm_1_abs_df = pd.concat(pm_1_abs_list)

pm_10_data_2018 = pm_10_data_2018.rename(columns = pm_10_column_names)
pm_10_data_2019 = pm_10_data_2019.rename(columns = pm_10_column_names)
pm_10_data_2020 = pm_10_data_2020.rename(columns = pm_10_column_names)
pm_10_data_2021 = pm_10_data_2021.rename(columns = pm_10_column_names)
pm_10_data_2022 = pm_10_data_2022.rename(columns = pm_10_column_names)

pm_10_data_2018.iloc[:, 0] = pd.to_datetime(pm_10_data_2018.iloc[:, 0])
pm_10_data_2019.iloc[:, 0] = pd.to_datetime(pm_10_data_2019.iloc[:, 0])
pm_10_data_2020.iloc[:, 0] = pd.to_datetime(pm_10_data_2020.iloc[:, 0])
pm_10_data_2021.iloc[:, 0] = pd.to_datetime(pm_10_data_2021.iloc[:, 0])
pm_10_data_2022.iloc[:, 0] = pd.to_datetime(pm_10_data_2022.iloc[:, 0])

pm_10_data_2018.iloc[:, 1:] = pm_10_data_2018.iloc[:, 1:].astype(float)
pm_10_data_2019.iloc[:, 1:] = pm_10_data_2019.iloc[:, 1:].astype(float)
pm_10_data_2020.iloc[:, 1:] = pm_10_data_2020.iloc[:, 1:].astype(float)
pm_10_data_2021.iloc[:, 1:] = pm_10_data_2021.iloc[:, 1:].astype(float)
pm_10_data_2022.iloc[:, 1:] = pm_10_data_2022.iloc[:, 1:].astype(float)

pm_10_data_2018 = pm_10_data_2018.set_index('date_and_time')
pm_10_data_2019 = pm_10_data_2019.set_index('date_and_time')
pm_10_data_2020 = pm_10_data_2020.set_index('date_and_time')
pm_10_data_2021 = pm_10_data_2021.set_index('date_and_time')
pm_10_data_2022 = pm_10_data_2022.set_index('date_and_time')

pm_10_data_2018 = pm_10_data_2018.resample('H', offset = '30min').mean()
pm_10_data_2019 = pm_10_data_2019.resample('H', offset = '30min').mean()
pm_10_data_2020 = pm_10_data_2020.resample('H', offset = '30min').mean()
pm_10_data_2021 = pm_10_data_2021.resample('H', offset = '30min').mean()
pm_10_data_2022 = pm_10_data_2022.resample('H', offset = '30min').mean()

pm_10_data_2018 = pm_10_data_2018.reset_index()
pm_10_data_2019 = pm_10_data_2019.reset_index()
pm_10_data_2020 = pm_10_data_2020.reset_index()
pm_10_data_2021 = pm_10_data_2021.reset_index()
pm_10_data_2022 = pm_10_data_2022.reset_index()

pm_10_data_2018['date_and_time'] = pd.to_datetime(pm_10_data_2018['date_and_time'])
pm_10_data_2018.iloc[:, 1:] = pm_10_data_2018.iloc[:, 1:].astype(float)
pm_10_data_2019['date_and_time'] = pd.to_datetime(pm_10_data_2019['date_and_time'])
pm_10_data_2019.iloc[:, 1:] = pm_10_data_2019.iloc[:, 1:].astype(float)
pm_10_data_2020['date_and_time'] = pd.to_datetime(pm_10_data_2020['date_and_time'])
pm_10_data_2020.iloc[:, 1:] = pm_10_data_2020.iloc[:, 1:].astype(float)
pm_10_data_2021['date_and_time'] = pd.to_datetime(pm_10_data_2021['date_and_time'])
pm_10_data_2021.iloc[:, 1:] = pm_10_data_2021.iloc[:, 1:].astype(float)
pm_10_data_2022['date_and_time'] = pd.to_datetime(pm_10_data_2022['date_and_time'])
pm_10_data_2022.iloc[:, 1:] = pm_10_data_2022.iloc[:, 1:].astype(float)

pm_10_data = pd.concat([pm_10_data_2018, pm_10_data_2019, pm_10_data_2020, pm_10_data_2021, pm_10_data_2022])
pm_10_data = pm_10_data.iloc[:, :8]

pm_10_abs_list = []
pm_10_abs_list.append(pm_10_abs_df)
pm_10_abs_list.append(pm_10_data)
pm_10_abs_df = pd.concat(pm_10_abs_list)

def process_year_data(directory_path, year):
    file_names = [file for file in os.listdir(directory_path) if file.endswith('.FLW')]

    data = pd.DataFrame(columns = ['date_and_time', 'dmps_file_relative_humidity', 'dmps_flw_file_temperature'])

    for file_name in file_names:

        month = int(file_name[4:6])
        day = int(file_name[6:8])

        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path, delim_whitespace = True, usecols=[0, 10, 11], header = None)

        df[0] = df.iloc[:, 0].apply(lambda x: convert_timestamp(year, month, day, int(x), str(x - int(x))[2:]))

        df.columns = ['date_and_time', 'dmps_file_relative_humidity', 'dmps_flw_file_temperature']

        data = pd.concat([data, df], ignore_index = True)

    data['date_and_time'] = pd.to_datetime(data['date_and_time'])

    return data

base_directory = r'./data'
years = [2018, 2019, 2020, 2021, 2022]

all_data = []
for year in years:
    year_directory = os.path.join(base_directory, f'Particle{str(year)[2:]}')
    year_data = process_year_data(year_directory, year)
    all_data.append(year_data)

combined_data = pd.concat(all_data, ignore_index = True)

combined_data = combined_data.sort_values('date_and_time')

combined_data.iloc[:, 0] = pd.to_datetime(combined_data.iloc[:, 0])

combined_data.iloc[:, 1:] = combined_data.iloc[:, 1:].astype(float)

combined_data = combined_data.set_index('date_and_time')

combined_data = combined_data.resample('H', offset = '30min').mean()

combined_data = combined_data.reset_index()

pm_rh_tp_2018 = pd.read_csv(r'./data\sca\smr_rhtp_20180101.csv')
pm_rh_tp_2019 = pd.read_csv(r'./data\sca\smr_rhtp_20190101.csv')
pm_rh_tp_2020 = pd.read_csv(r'./data\sca\smr_rhtp_20200101.csv')
pm_rh_tp_2021 = pd.read_csv(r'./data\sca\smr_rhtp_20210101.csv')
pm_rh_tp_2022 = pd.read_csv(r'./data\sca\smr_rhtp_20220101.csv')

def extract_rhtp_data(df_year):
    df_year = df_year.iloc[1:]

    df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'] = pd.to_datetime(
        df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'], format='%Y-%m-%d %H:%M:%S'
    )

    pm_1_range = [(3, 9), (23, 29), (43, 49)]
    pm_10_range = [(13, 19), (33, 39), (53, 59)]

    pm_1_rh_tp = df_year[df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'].dt.minute.isin(
        [minute for start, end in pm_1_range for minute in range(start, end + 1)]
    )]

    pm_10_rh_tp = df_year[df_year['Date String (YYYY-MM-DD hh:mm:ss) UTC'].dt.minute.isin(
        [minute for start, end in pm_10_range for minute in range(start, end + 1)]
    )]

    return pm_1_rh_tp, pm_10_rh_tp

rh_tp_1_2018, rh_tp_10_2018 = extract_rhtp_data(pm_rh_tp_2018)
rh_tp_1_2019, rh_tp_10_2019 = extract_rhtp_data(pm_rh_tp_2019)
rh_tp_1_2020, rh_tp_10_2020 = extract_rhtp_data(pm_rh_tp_2020)
rh_tp_1_2021, rh_tp_10_2021 = extract_rhtp_data(pm_rh_tp_2021)
rh_tp_1_2022, rh_tp_10_2022 = extract_rhtp_data(pm_rh_tp_2022)

rh_tp_1_2018 = pd.concat([rh_tp_1_2018.iloc[:, [0]], rh_tp_1_2018.iloc[:, -3:]], axis = 1)
rh_tp_1_2019 = pd.concat([rh_tp_1_2019.iloc[:, [0]], rh_tp_1_2019.iloc[:, -3:]], axis = 1)
rh_tp_1_2020 = pd.concat([rh_tp_1_2020.iloc[:, [0]], rh_tp_1_2020.iloc[:, -3:]], axis = 1)
rh_tp_1_2021 = pd.concat([rh_tp_1_2021.iloc[:, [0]], rh_tp_1_2021.iloc[:, -3:]], axis = 1)
rh_tp_1_2022 = pd.concat([rh_tp_1_2022.iloc[:, [0]], rh_tp_1_2022.iloc[:, -3:]], axis = 1)

rh_tp_10_2018 = pd.concat([rh_tp_10_2018.iloc[:, [0]], rh_tp_10_2018.iloc[:, -3:]], axis = 1)
rh_tp_10_2019 = pd.concat([rh_tp_10_2019.iloc[:, [0]], rh_tp_10_2019.iloc[:, -3:]], axis = 1)
rh_tp_10_2020 = pd.concat([rh_tp_10_2020.iloc[:, [0]], rh_tp_10_2020.iloc[:, -3:]], axis = 1)
rh_tp_10_2021 = pd.concat([rh_tp_10_2021.iloc[:, [0]], rh_tp_10_2021.iloc[:, -3:]], axis = 1)
rh_tp_10_2022 = pd.concat([rh_tp_10_2022.iloc[:, [0]], rh_tp_10_2022.iloc[:, -3:]], axis = 1)

rh_tp_1_columns = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Sample temperature (°C)': 'sample_temperature',
    'Sample RH (%)': 'relative_humidity',
    'Sample pressure (hPa)': 'sample_pressure'    
}

rh_tp_10_columns = {
    'Date String (YYYY-MM-DD hh:mm:ss) UTC': 'date_and_time',
    'Sample temperature (°C)': 'sample_temperature',
    'Sample RH (%)': 'relative_humidity',
    'Sample pressure (hPa)': 'sample_pressure'    
}

rh_tp_1_2018 = rh_tp_1_2018.rename(columns = rh_tp_1_columns)
rh_tp_1_2019 = rh_tp_1_2019.rename(columns = rh_tp_1_columns)
rh_tp_1_2020 = rh_tp_1_2020.rename(columns = rh_tp_1_columns)
rh_tp_1_2021 = rh_tp_1_2021.rename(columns = rh_tp_1_columns)
rh_tp_1_2022 = rh_tp_1_2022.rename(columns = rh_tp_1_columns)

rh_tp_1_2018.iloc[:, 0] = pd.to_datetime(rh_tp_1_2018.iloc[:, 0])
rh_tp_1_2019.iloc[:, 0] = pd.to_datetime(rh_tp_1_2019.iloc[:, 0])
rh_tp_1_2020.iloc[:, 0] = pd.to_datetime(rh_tp_1_2020.iloc[:, 0])
rh_tp_1_2021.iloc[:, 0] = pd.to_datetime(rh_tp_1_2021.iloc[:, 0])
rh_tp_1_2022.iloc[:, 0] = pd.to_datetime(rh_tp_1_2022.iloc[:, 0])

rh_tp_1_2018.iloc[:, 1:] = rh_tp_1_2018.iloc[:, 1:].astype(float)
rh_tp_1_2019.iloc[:, 1:] = rh_tp_1_2019.iloc[:, 1:].astype(float)
rh_tp_1_2020.iloc[:, 1:] = rh_tp_1_2020.iloc[:, 1:].astype(float)
rh_tp_1_2021.iloc[:, 1:] = rh_tp_1_2021.iloc[:, 1:].astype(float)
rh_tp_1_2022.iloc[:, 1:] = rh_tp_1_2022.iloc[:, 1:].astype(float)

rh_tp_1_2018 = rh_tp_1_2018.set_index('date_and_time')
rh_tp_1_2019 = rh_tp_1_2019.set_index('date_and_time')
rh_tp_1_2020 = rh_tp_1_2020.set_index('date_and_time')
rh_tp_1_2021 = rh_tp_1_2021.set_index('date_and_time')
rh_tp_1_2022 = rh_tp_1_2022.set_index('date_and_time')

rh_tp_1_2018 = rh_tp_1_2018.resample('H', offset = '30min').mean()
rh_tp_1_2019 = rh_tp_1_2019.resample('H', offset = '30min').mean()
rh_tp_1_2020 = rh_tp_1_2020.resample('H', offset = '30min').mean()
rh_tp_1_2021 = rh_tp_1_2021.resample('H', offset = '30min').mean()
rh_tp_1_2022 = rh_tp_1_2022.resample('H', offset = '30min').mean()

rh_tp_1_2018 = rh_tp_1_2018.reset_index()
rh_tp_1_2019 = rh_tp_1_2019.reset_index()
rh_tp_1_2020 = rh_tp_1_2020.reset_index()
rh_tp_1_2021 = rh_tp_1_2021.reset_index()
rh_tp_1_2022 = rh_tp_1_2022.reset_index()

rh_tp_10_2018 = rh_tp_10_2018.rename(columns = rh_tp_10_columns)
rh_tp_10_2019 = rh_tp_10_2019.rename(columns = rh_tp_10_columns)
rh_tp_10_2020 = rh_tp_10_2020.rename(columns = rh_tp_10_columns)
rh_tp_10_2021 = rh_tp_10_2021.rename(columns = rh_tp_10_columns)
rh_tp_10_2022 = rh_tp_10_2022.rename(columns = rh_tp_10_columns)

rh_tp_10_2018.iloc[:, 0] = pd.to_datetime(rh_tp_10_2018.iloc[:, 0])
rh_tp_10_2019.iloc[:, 0] = pd.to_datetime(rh_tp_10_2019.iloc[:, 0])
rh_tp_10_2020.iloc[:, 0] = pd.to_datetime(rh_tp_10_2020.iloc[:, 0])
rh_tp_10_2021.iloc[:, 0] = pd.to_datetime(rh_tp_10_2021.iloc[:, 0])
rh_tp_10_2022.iloc[:, 0] = pd.to_datetime(rh_tp_10_2022.iloc[:, 0])

rh_tp_10_2018.iloc[:, 1:] = rh_tp_10_2018.iloc[:, 1:].astype(float)
rh_tp_10_2019.iloc[:, 1:] = rh_tp_10_2019.iloc[:, 1:].astype(float)
rh_tp_10_2020.iloc[:, 1:] = rh_tp_10_2020.iloc[:, 1:].astype(float)
rh_tp_10_2021.iloc[:, 1:] = rh_tp_10_2021.iloc[:, 1:].astype(float)
rh_tp_10_2022.iloc[:, 1:] = rh_tp_10_2022.iloc[:, 1:].astype(float)

rh_tp_10_2018 = rh_tp_10_2018.set_index('date_and_time')
rh_tp_10_2019 = rh_tp_10_2019.set_index('date_and_time')
rh_tp_10_2020 = rh_tp_10_2020.set_index('date_and_time')
rh_tp_10_2021 = rh_tp_10_2021.set_index('date_and_time')
rh_tp_10_2022 = rh_tp_10_2022.set_index('date_and_time')

rh_tp_10_2019 = rh_tp_10_2019.resample('H', offset = '30min').mean()
rh_tp_10_2020 = rh_tp_10_2020.resample('H', offset = '30min').mean()
rh_tp_10_2021 = rh_tp_10_2021.resample('H', offset = '30min').mean()
rh_tp_10_2022 = rh_tp_10_2022.resample('H', offset = '30min').mean()

rh_tp_10_2018 = rh_tp_10_2018.reset_index()
rh_tp_10_2019 = rh_tp_10_2019.reset_index()
rh_tp_10_2020 = rh_tp_10_2020.reset_index()
rh_tp_10_2021 = rh_tp_10_2021.reset_index()
rh_tp_10_2022 = rh_tp_10_2022.reset_index()

rh_tp_1_2018['date_and_time'] = pd.to_datetime(rh_tp_1_2018['date_and_time'])
rh_tp_1_2018.iloc[:, 1:] = rh_tp_1_2018.iloc[:, 1:].astype(float)
rh_tp_1_2018 = rh_tp_1_2018.sort_values('date_and_time')
rh_tp_1_2019['date_and_time'] = pd.to_datetime(rh_tp_1_2019['date_and_time'])
rh_tp_1_2019.iloc[:, 1:] = rh_tp_1_2019.iloc[:, 1:].astype(float)
rh_tp_1_2019 = rh_tp_1_2019.sort_values('date_and_time')
rh_tp_1_2020['date_and_time'] = pd.to_datetime(rh_tp_1_2020['date_and_time'])
rh_tp_1_2020.iloc[:, 1:] = rh_tp_1_2020.iloc[:, 1:].astype(float)
rh_tp_1_2020 = rh_tp_1_2020.sort_values('date_and_time')
rh_tp_1_2021['date_and_time'] = pd.to_datetime(rh_tp_1_2021['date_and_time'])
rh_tp_1_2021.iloc[:, 1:] = rh_tp_1_2021.iloc[:, 1:].astype(float)
rh_tp_1_2021 = rh_tp_1_2021.sort_values('date_and_time')
rh_tp_1_2022['date_and_time'] = pd.to_datetime(rh_tp_1_2022['date_and_time'])
rh_tp_1_2022.iloc[:, 1:] = rh_tp_1_2022.iloc[:, 1:].astype(float)
rh_tp_1_2022 = rh_tp_1_2022.sort_values('date_and_time')

rh_tp_1_2018 = pd.merge_asof(rh_tp_1_2018, combined_data, on = 'date_and_time')
rh_tp_1_2019 = pd.merge_asof(rh_tp_1_2019, combined_data, on = 'date_and_time')
rh_tp_1_2020 = pd.merge_asof(rh_tp_1_2020, combined_data, on = 'date_and_time')
rh_tp_1_2021 = pd.merge_asof(rh_tp_1_2021, combined_data, on = 'date_and_time')
rh_tp_1_2022 = pd.merge_asof(rh_tp_1_2022, combined_data, on = 'date_and_time')

rh_tp_1_2018['correction_factor'] = 0.2697/((rh_tp_1_2018['dmps_flw_file_temperature'] + 273.15)/rh_tp_1_2018['sample_pressure'])
rh_tp_1_2019['correction_factor'] = 0.2697/((rh_tp_1_2019['dmps_flw_file_temperature'] + 273.15)/rh_tp_1_2019['sample_pressure'])
rh_tp_1_2020['correction_factor'] = 0.2697/((rh_tp_1_2020['dmps_flw_file_temperature'] + 273.15)/rh_tp_1_2020['sample_pressure'])
rh_tp_1_2021['correction_factor'] = 0.2697/((rh_tp_1_2021['dmps_flw_file_temperature'] + 273.15)/rh_tp_1_2021['sample_pressure'])
rh_tp_1_2022['correction_factor'] = 0.2697/((rh_tp_1_2022['dmps_flw_file_temperature'] + 273.15)/rh_tp_1_2022['sample_pressure'])

pm_1_data = pd.concat([rh_tp_1_2018, rh_tp_1_2019, rh_tp_1_2020, rh_tp_1_2021, rh_tp_1_2022])
pm_1_data.sort_values('date_and_time', inplace = True)
pm_1_abs_df.sort_values('date_and_time', inplace = True)
pm_1_abs_df = pd.merge_asof(pm_1_abs_df, pm_1_data, on = 'date_and_time', direction = 'nearest')
date_and_time_range = pm_1_abs_df['date_and_time'].dt.year <= 2017
pm_1_abs_df.loc[(date_and_time_range) | (pd.isna(pm_1_abs_df['correction_factor'])), 'correction_factor'] = 1

update_year_list = [2018, 2019, 2020, 2021, 2022]
update_condition = pm_1_abs_df['date_and_time'].dt.year.isin(update_year_list)

slope = 2.33
intercept = -0.16

pm_1_abs_df.loc[update_condition, 'pm_1_abs_370'] = (pm_1_abs_df['pm_1_abs_370'] + intercept)/slope
pm_1_abs_df.loc[update_condition, 'pm_1_abs_470'] = (pm_1_abs_df['pm_1_abs_470'] + intercept)/slope
pm_1_abs_df.loc[update_condition, 'pm_1_abs_520'] = (pm_1_abs_df['pm_1_abs_520'] + intercept)/slope
pm_1_abs_df.loc[update_condition, 'pm_1_abs_590'] = (pm_1_abs_df['pm_1_abs_590'] + intercept)/slope
pm_1_abs_df.loc[update_condition, 'pm_1_abs_660'] = (pm_1_abs_df['pm_1_abs_660'] + intercept)/slope
pm_1_abs_df.loc[update_condition, 'pm_1_abs_880'] = (pm_1_abs_df['pm_1_abs_880'] + intercept)/slope
pm_1_abs_df.loc[update_condition, 'pm_1_abs_950'] = (pm_1_abs_df['pm_1_abs_950'] + intercept)/slope

pm_1_abs_df.loc[update_condition, 'pm_1_abs_370'] = pm_1_abs_df['pm_1_abs_370'] * pm_1_abs_df['correction_factor']
pm_1_abs_df.loc[update_condition, 'pm_1_abs_470'] = pm_1_abs_df['pm_1_abs_470'] * pm_1_abs_df['correction_factor']
pm_1_abs_df.loc[update_condition, 'pm_1_abs_520'] = pm_1_abs_df['pm_1_abs_520'] * pm_1_abs_df['correction_factor']
pm_1_abs_df.loc[update_condition, 'pm_1_abs_590'] = pm_1_abs_df['pm_1_abs_590'] * pm_1_abs_df['correction_factor']
pm_1_abs_df.loc[update_condition, 'pm_1_abs_660'] = pm_1_abs_df['pm_1_abs_660'] * pm_1_abs_df['correction_factor']
pm_1_abs_df.loc[update_condition, 'pm_1_abs_880'] = pm_1_abs_df['pm_1_abs_880'] * pm_1_abs_df['correction_factor']
pm_1_abs_df.loc[update_condition, 'pm_1_abs_950'] = pm_1_abs_df['pm_1_abs_950'] * pm_1_abs_df['correction_factor']

pm_10_abs_df = pm_10_abs_df[(pm_10_abs_df['date_and_time'].dt.month != 7) | (pm_10_abs_df['date_and_time'].dt.year != 2011)]

pm_10_abs_df = pm_10_abs_df[(pm_10_abs_df['date_and_time'].dt.month != 1) | (pm_10_abs_df['date_and_time'].dt.year != 2018)]

pm_10_abs_df = pm_10_abs_df[(pm_10_abs_df['date_and_time'].dt.month != 2) | (pm_10_abs_df['date_and_time'].dt.year != 2018)]

columns_to_check = ['pm_1_abs_520']

for col in columns_to_check:
    mask = pm_1_abs_df[col] != pm_1_abs_df[col].shift(1)

    group_id = mask.cumsum()

    counts = pm_1_abs_df.groupby(group_id).size()

    groups_to_remove = counts[counts >= 5].index

    pm_1_abs_df = pm_1_abs_df[~group_id.isin(groups_to_remove)]

pm_1_abs_df = pm_1_abs_df.query('pm_1_abs_520 >= 0.05')

pm_1_abs_df = pm_1_abs_df[pm_1_abs_df['dmps_file_relative_humidity'] <= 40]

pm_1_abs_df = pm_1_abs_df.iloc[:, :8]

rh_tp_10_2018['date_and_time'] = pd.to_datetime(rh_tp_10_2018['date_and_time'])
rh_tp_10_2018.iloc[:, 1:] = rh_tp_10_2018.iloc[:, 1:].astype(float)
rh_tp_10_2018 = rh_tp_10_2018.sort_values('date_and_time')
rh_tp_10_2019['date_and_time'] = pd.to_datetime(rh_tp_10_2019['date_and_time'])
rh_tp_10_2019.iloc[:, 1:] = rh_tp_10_2019.iloc[:, 1:].astype(float)
rh_tp_10_2018 = rh_tp_10_2018.sort_values('date_and_time')
rh_tp_10_2020['date_and_time'] = pd.to_datetime(rh_tp_10_2020['date_and_time'])
rh_tp_10_2020.iloc[:, 1:] = rh_tp_10_2020.iloc[:, 1:].astype(float)
rh_tp_10_2018 = rh_tp_10_2018.sort_values('date_and_time')
rh_tp_10_2021['date_and_time'] = pd.to_datetime(rh_tp_10_2021['date_and_time'])
rh_tp_10_2021.iloc[:, 1:] = rh_tp_10_2021.iloc[:, 1:].astype(float)
rh_tp_10_2018 = rh_tp_10_2018.sort_values('date_and_time')
rh_tp_10_2022['date_and_time'] = pd.to_datetime(rh_tp_10_2022['date_and_time'])
rh_tp_10_2022.iloc[:, 1:] = rh_tp_10_2022.iloc[:, 1:].astype(float)
rh_tp_10_2018 = rh_tp_10_2018.sort_values('date_and_time')

rh_tp_10_2018 = pd.merge_asof(rh_tp_10_2018, combined_data, on = 'date_and_time')
rh_tp_10_2019 = pd.merge_asof(rh_tp_10_2019, combined_data, on = 'date_and_time')
rh_tp_10_2020 = pd.merge_asof(rh_tp_10_2020, combined_data, on = 'date_and_time')
rh_tp_10_2021 = pd.merge_asof(rh_tp_10_2021, combined_data, on = 'date_and_time')
rh_tp_10_2022 = pd.merge_asof(rh_tp_10_2022, combined_data, on = 'date_and_time')

rh_tp_10_2018['correction_factor'] = 0.2697/((rh_tp_10_2018['dmps_flw_file_temperature'] + 273.15)/rh_tp_10_2018['sample_pressure'])
rh_tp_10_2019['correction_factor'] = 0.2697/((rh_tp_10_2019['dmps_flw_file_temperature'] + 273.15)/rh_tp_10_2019['sample_pressure'])
rh_tp_10_2020['correction_factor'] = 0.2697/((rh_tp_10_2020['dmps_flw_file_temperature'] + 273.15)/rh_tp_10_2020['sample_pressure'])
rh_tp_10_2021['correction_factor'] = 0.2697/((rh_tp_10_2021['dmps_flw_file_temperature'] + 273.15)/rh_tp_10_2021['sample_pressure'])
rh_tp_10_2022['correction_factor'] = 0.2697/((rh_tp_10_2022['dmps_flw_file_temperature'] + 273.15)/rh_tp_10_2022['sample_pressure'])

pm_10_data = pd.concat([rh_tp_10_2018, rh_tp_10_2019, rh_tp_10_2020, rh_tp_10_2021, rh_tp_10_2022])
pm_10_data.sort_values('date_and_time', inplace = True)
pm_10_abs_df.sort_values('date_and_time', inplace = True)
pm_10_abs_df = pd.merge_asof(pm_10_abs_df, pm_10_data, on = 'date_and_time', direction = 'nearest')
date_and_time_range = pm_10_abs_df['date_and_time'].dt.year <= 2017
pm_10_abs_df.loc[(date_and_time_range) | (pd.isna(pm_10_abs_df['correction_factor'])), 'correction_factor'] = 1

update_year_list = [2018, 2019, 2020, 2021, 2022]
update_condition = pm_10_abs_df['date_and_time'].dt.year.isin(update_year_list)

slope = 2.33
intercept = -0.16

pm_10_abs_df.loc[update_condition, 'pm_10_abs_370'] = (pm_10_abs_df['pm_10_abs_370'] + intercept)/slope
pm_10_abs_df.loc[update_condition, 'pm_10_abs_470'] = (pm_10_abs_df['pm_10_abs_470'] + intercept)/slope
pm_10_abs_df.loc[update_condition, 'pm_10_abs_520'] = (pm_10_abs_df['pm_10_abs_520'] + intercept)/slope
pm_10_abs_df.loc[update_condition, 'pm_10_abs_590'] = (pm_10_abs_df['pm_10_abs_590'] + intercept)/slope
pm_10_abs_df.loc[update_condition, 'pm_10_abs_660'] = (pm_10_abs_df['pm_10_abs_660'] + intercept)/slope
pm_10_abs_df.loc[update_condition, 'pm_10_abs_880'] = (pm_10_abs_df['pm_10_abs_880'] + intercept)/slope
pm_10_abs_df.loc[update_condition, 'pm_10_abs_950'] = (pm_10_abs_df['pm_10_abs_950'] + intercept)/slope

pm_10_abs_df.loc[update_condition, 'pm_10_abs_370'] = pm_10_abs_df['pm_10_abs_370'] * pm_10_abs_df['correction_factor']
pm_10_abs_df.loc[update_condition, 'pm_10_abs_470'] = pm_10_abs_df['pm_10_abs_470'] * pm_10_abs_df['correction_factor']
pm_10_abs_df.loc[update_condition, 'pm_10_abs_520'] = pm_10_abs_df['pm_10_abs_520'] * pm_10_abs_df['correction_factor']
pm_10_abs_df.loc[update_condition, 'pm_10_abs_590'] = pm_10_abs_df['pm_10_abs_590'] * pm_10_abs_df['correction_factor']
pm_10_abs_df.loc[update_condition, 'pm_10_abs_660'] = pm_10_abs_df['pm_10_abs_660'] * pm_10_abs_df['correction_factor']
pm_10_abs_df.loc[update_condition, 'pm_10_abs_880'] = pm_10_abs_df['pm_10_abs_880'] * pm_10_abs_df['correction_factor']
pm_10_abs_df.loc[update_condition, 'pm_10_abs_950'] = pm_10_abs_df['pm_10_abs_950'] * pm_10_abs_df['correction_factor']

pm_1_abs_df = pm_1_abs_df[(pm_1_abs_df['date_and_time'].dt.month != 7) | (pm_1_abs_df['date_and_time'].dt.year != 2011)]

pm_1_abs_df = pm_1_abs_df[(pm_1_abs_df['date_and_time'].dt.month != 1) | (pm_1_abs_df['date_and_time'].dt.year != 2018)]

pm_1_abs_df = pm_1_abs_df[(pm_1_abs_df['date_and_time'].dt.month != 2) | (pm_1_abs_df['date_and_time'].dt.year != 2018)]

columns_to_check = ['pm_10_abs_520']

for col in columns_to_check:
    mask = pm_10_abs_df[col] != pm_10_abs_df[col].shift(1)

    group_id = mask.cumsum()

    counts = pm_10_abs_df.groupby(group_id).size()

    groups_to_remove = counts[counts >= 5].index

    pm_10_abs_df = pm_10_abs_df[~group_id.isin(groups_to_remove)]

pm_10_abs_df = pm_10_abs_df.query('pm_10_abs_520 >= 0.05') 

pm_10_abs_df = pm_10_abs_df[pm_10_abs_df['dmps_file_relative_humidity'] <= 40]

pm_10_abs_df = pm_10_abs_df.iloc[:, :8]

abs_10_1_df = pd.merge_asof(pm_10_abs_df, pm_1_abs_df, on='date_and_time')

abs_10_1_df['date_and_time'] = pd.to_datetime(abs_10_1_df['date_and_time'])

abs_10_1_df.iloc[:, 1:] = abs_10_1_df.iloc[:, 1:].astype(float)

wavelengths = [370, 470, 520, 590, 660, 880, 950]

for wl in wavelengths:
    pm_10_col = f'pm_10_abs_{wl}'
    pm_1_col = f'pm_1_abs_{wl}'
    abs_col = f'abs_10_1_{wl}'

    abs_10_1_df[abs_col] = abs_10_1_df[pm_10_col] - abs_10_1_df[pm_1_col]

    abs_10_1_df.loc[abs_10_1_df[pm_10_col].isna() | abs_10_1_df[pm_1_col].isna(), abs_col] = np.nan

abs_10_1_df['date_and_time'] = pd.to_datetime(abs_10_1_df['date_and_time'])
abs_10_1_df.iloc[:, 1:] = abs_10_1_df.iloc[:, 1:].astype(float)

start_date = pd.to_datetime('2010-10-04 00:00:00')
end_date = pd.to_datetime('2022-10-04 00:00:00')

abs_10_1_df = abs_10_1_df[(abs_10_1_df['date_and_time'] >= start_date) & 
                           (abs_10_1_df['date_and_time'] <= end_date)]

abs_10_1_df['date_only'] = abs_10_1_df['date_and_time'].dt.floor('D')

data_per_day = abs_10_1_df.groupby('date_only').count()['date_and_time']

valid_days = data_per_day[data_per_day >= 18].index

filtered_abs_10_1_df = abs_10_1_df[abs_10_1_df['date_only'].isin(valid_days)]

abs_10_1_df = filtered_abs_10_1_df.groupby('date_only').mean(numeric_only=True)

abs_10_1_df = abs_10_1_df.reset_index()

abs_10_1_df = abs_10_1_df.rename(columns={'date_only': 'date_and_time'})

abs_10_1_df['date_and_time'] = abs_10_1_df['date_and_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

abs_10_1_df['date_and_time'] = pd.to_datetime(abs_10_1_df['date_and_time'])

pm_10_aae_df = abs_10_1_df[['date_and_time', 'pm_10_abs_370', 'pm_10_abs_470', 'pm_10_abs_520', 'pm_10_abs_590', 'pm_10_abs_660', 'pm_10_abs_880', 'pm_10_abs_950']].copy()
pm_10_aae_df['date_and_time'] = pd.to_datetime(pm_10_aae_df['date_and_time'], errors='coerce')

pm_1_aae_df = abs_10_1_df[['date_and_time', 'pm_1_abs_370', 'pm_1_abs_470', 'pm_1_abs_520', 'pm_1_abs_590', 'pm_1_abs_660', 'pm_1_abs_880', 'pm_1_abs_950']].copy()
pm_1_aae_df['date_and_time'] = pd.to_datetime(pm_1_aae_df['date_and_time'], errors='coerce')

aae_10_1_df = abs_10_1_df[['date_and_time', 'abs_10_1_370', 'abs_10_1_470', 'abs_10_1_520', 'abs_10_1_590', 'abs_10_1_660', 'abs_10_1_880', 'abs_10_1_950']].copy()
aae_10_1_df['date_and_time'] = pd.to_datetime(aae_10_1_df['date_and_time'], errors='coerce')

def custom_aae_round(value):
    value_str = f"{value:.3f}"
    last_digit = int(value_str[-1])

    if last_digit in [0, 1, 2, 3, 4]:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = Decimal(value_str).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP) + Decimal("0.01")

    return float(rounded_value)

def generate_aae_y_ticks(y_limits):
    y_min, y_max = y_limits
    y_ticks_raw = np.linspace(y_min, y_max, num=6)
    y_ticks = [custom_aae_round(tick) for tick in y_ticks_raw]
    return y_ticks

def format_aae_number(value, decimals=2):
    if abs(value) >= 100 or (abs(value) < 0.01 and value != 0):
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10**exponent)
        return f"{coefficient:.{decimals}f} × 10$^{{{exponent}}}$"
    else:
        quantization_format = f"1.{'0' * decimals}"
        return f"{Decimal(str(value)).quantize(Decimal(quantization_format), rounding=ROUND_HALF_UP):.{decimals}f}"

def format_aae_yaxis(value, _):
    return f"{value:.2f}"

def compute_aae(df, name):
    if 'date_and_time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'date_and_time' column.")

    numeric_cols = [col for col in df.columns if col != 'date_and_time']

    df_numeric = df[numeric_cols].copy().where(df[numeric_cols] > 0)

    wavelengths = np.array([370, 470, 520, 590, 660, 880, 950])
    log_lambda = np.log(wavelengths).reshape(-1, 1)

    valid_rows = df_numeric.dropna().astype(float)

    log_sigma = np.log(valid_rows)

    if log_sigma.shape[0] < 2:
        raise ValueError("Not enough valid rows for regression.")

    X = sm.add_constant(log_lambda)
    Y = log_sigma.values.T

    model = sm.OLS(Y, X).fit()
    slopes = model.params[1, :]

    aae_df = df[['date_and_time']].copy()
    aae_df[name] = np.nan
    aae_df.loc[valid_rows.index, name] = -slopes

    return aae_df

pm_10_aae_df = compute_aae(pm_10_aae_df, name='aae_0')

pm_1_aae_df = compute_aae(pm_1_aae_df, name='aae_1')

aae_10_1_df = compute_aae(aae_10_1_df, name='aae_2')

def plot_aae_trend(x, y, filename):
    """
    Plots the Scattering Ångström Exponent (SAE) trend with Mann-Kendall analysis.

    Parameters:
        x (pd.Series): Datetime index for the x-axis.
        y (pd.Series): SAE values for the y-axis.
        filename (str): Name of the file to save the plot.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(30, 10))

    y = y.replace([np.inf, -np.inf], np.nan)
    valid_index = y.dropna().index
    x = x.loc[valid_index]
    y = y.loc[valid_index]

    unique_values = y.nunique()
    tau, p_value = np.nan, np.nan

    if unique_values > 1:
        try:
            x_num = mdates.date2num(x)
            tau, p_value = kendalltau(x_num, y)
        except Exception as e:
            print(f"Error in Mann-Kendall test: {e}")

    p_value_text = f"$p$-value: {format_aae_number(p_value, 2)}" if np.isfinite(p_value) else r"$p$-value: N/A"

    x_segments = np.split(x, np.where(np.diff(x) > np.timedelta64(45, 'D'))[0] + 1)
    y_segments = np.split(y, np.where(np.diff(x) > np.timedelta64(45, 'D'))[0] + 1)

    for x_seg, y_seg in zip(x_segments, y_segments):
        ax.plot(x_seg, y_seg, marker='o', markersize=20, linestyle='-', color='cornflowerblue', linewidth=6, alpha=1)

    try:
        if unique_values > 1:
            slope, intercept = np.polyfit(mdates.date2num(x), y, 1)
            trendline = np.poly1d([slope, intercept])
            linestyle = '-' if p_value <= 0.05 else '--'
            ax.plot(x, trendline(mdates.date2num(x)), 'red', linestyle=linestyle, alpha=1, linewidth=6)

            slope_per_year = slope * 12  
            slope_text = rf"Slope: {format_aae_number(slope_per_year, 2)} yr$^{{-1}}$"
            slope_percent = ((slope / y.median()) * 12) * 100
            slope_percent_text = rf"Relative trend: {format_aae_number(slope_percent, 2)} %yr$^{{-1}}$"

            ax.text(0.03, 0.95, p_value_text, ha='left', va='top', transform=ax.transAxes, fontsize=60, color='black')
            ax.text(0.03, 0.77, slope_text, ha='left', va='top', transform=ax.transAxes, fontsize=60, color='black')
            ax.text(0.03, 0.59, slope_percent_text, ha='left', va='top', transform=ax.transAxes, fontsize=60, color='black')

    except np.linalg.LinAlgError:
        print(f"Skipping trendline fitting due to numerical issues.")

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax.set_yticks(generate_aae_y_ticks(ax.get_ylim()))

    ax.grid(True, linestyle='--', linewidth=3, color='gray', alpha=1)
    ax.tick_params(axis='x', direction='out', colors='black', width=6, length=18, labelsize=60)
    ax.tick_params(axis='y', direction='out', colors='black', width=6, length=18, labelsize=60)

    for spine in ax.spines.values():
        spine.set_linewidth(6)

    ax.set_xlabel("Year", fontsize=60)
    ax.set_ylabel("AAE (dimensionless)", fontsize=60)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

plot_aae_trend(pm_10_aae_df['date_and_time'], pm_10_aae_df['aae_0'], 'aae_pm_10_plot.png')
plot_aae_trend(pm_1_aae_df['date_and_time'], pm_1_aae_df['aae_1'], 'aae_pm_1_plot.png')
plot_aae_trend(aae_10_1_df['date_and_time'], aae_10_1_df['aae_2'], 'aae_10_1_plot.png')

dust_event = pd.read_csv(r'./data\dust_events.csv')

dust_event['date_and_time'] = pd.to_datetime(dust_event['date_and_time'])

dust_event['dust_event'] = 1

dust_event['date_and_time'] = dust_event['date_and_time'].dt.normalize()

aae_merged_df = (
    pm_10_aae_df.merge(pm_1_aae_df, on='date_and_time', how='outer')
    .merge(aae_10_1_df, on='date_and_time', how='outer')
)

aae_0 = pm_10_aae_df
aae_1 = pm_1_aae_df
aae_2 = aae_10_1_df

aae_0.sort_values(by='date_and_time', inplace=True)
aae_1.sort_values(by='date_and_time', inplace=True)
aae_2.sort_values(by='date_and_time', inplace=True)

abs_10_1_df = abs_10_1_df.drop(columns=['aae_0', 'aae_1', 'aae_2'], errors='ignore')

abs_10_1_df = abs_10_1_df.merge(aae_0, on='date_and_time', how='left')
abs_10_1_df = abs_10_1_df.merge(aae_1, on='date_and_time', how='left')
abs_10_1_df = abs_10_1_df.merge(aae_2, on='date_and_time', how='left')

common_cols = set(sca_10_1_df.columns) & set(abs_10_1_df.columns) - {'date_and_time'}
sca_abs_df = pd.merge(
    sca_10_1_df, abs_10_1_df,
    on='date_and_time', how='outer', suffixes=('', '_dup')
)

sca_abs_df.drop(columns=[col + '_dup' for col in common_cols if col + '_dup' in sca_abs_df.columns], inplace=True)

sca_abs_df['date_and_time'] = pd.to_datetime(sca_abs_df['date_and_time'])
sca_abs_df.iloc[:, 1:] = sca_abs_df.iloc[:, 1:].astype(float)
sca_abs_df.sort_values(by='date_and_time', inplace=True)

def calculate_cazorla_2013_aae(df, col1, col2, new_col_name, lambda1=470, lambda2=660):
    df[new_col_name] = -np.log(df[col1] / df[col2]) / np.log(lambda1 / lambda2)
    return df

def calculate_cazorla_2013_sae(df, col1, col2, new_col_name, lambda1=450, lambda2=700):
    df[new_col_name] = -np.log(df[col1] / df[col2]) / np.log(lambda1 / lambda2)
    return df

sca_abs_df = calculate_cazorla_2013_aae(
    sca_abs_df, 'pm_10_abs_470', 'pm_10_abs_660', 'cazorla_2013_aae_0'
)

sca_abs_df = calculate_cazorla_2013_aae(
    sca_abs_df, 'pm_1_abs_470', 'pm_1_abs_660', 'cazorla_2013_aae_1'
)

sca_abs_df = calculate_cazorla_2013_aae(
    sca_abs_df, 'abs_10_1_470', 'abs_10_1_660', 'cazorla_2013_aae_2'
)

sca_abs_df = calculate_cazorla_2013_sae(
    sca_abs_df, 'pm_10_sca_450', 'pm_10_sca_700', 'cazorla_2013_sae_0'
)

sca_and_df = calculate_cazorla_2013_sae(
    sca_abs_df, 'pm_1_sca_450', 'pm_1_sca_700', 'cazorla_2013_sae_1'
)

sca_abs_df = calculate_cazorla_2013_sae(
    sca_abs_df, 'sca_10_1_450', 'sca_10_1_700', 'cazorla_2013_sae_2'
)

sca_abs_df['pm_10_abs_550'] = np.exp(
    sca_abs_df['aae_0'] * np.log(550 / 520) +
    np.log(sca_abs_df['pm_10_abs_520'])
)

sca_abs_df['pm_1_abs_550'] = np.exp(
    sca_abs_df['aae_1'] * np.log(550 / 520) +
    np.log(sca_abs_df['pm_1_abs_520'])
)

sca_abs_df['abs_10_1_550'] = np.exp(
    sca_abs_df['aae_2'] * np.log(550 / 520) +
    np.log(sca_abs_df['abs_10_1_520'])
)

sca_abs_df['pm_10_ssa_550'] = sca_abs_df['pm_10_sca_550'] / (
    sca_abs_df['pm_10_sca_550'] + sca_abs_df['pm_10_abs_550']
)

sca_abs_df['pm_1_ssa_550'] = sca_abs_df['pm_1_sca_550'] / (
    sca_abs_df['pm_1_sca_550'] + sca_abs_df['pm_1_abs_550']
)

sca_abs_df['ssa_10_1_550'] = sca_abs_df['sca_10_1_550'] / (
    sca_abs_df['sca_10_1_550'] + sca_abs_df['abs_10_1_550']
)

sca_abs_dust_df = copy.deepcopy(sca_abs_df)

sca_abs_dust_df = pd.merge(sca_abs_dust_df, dust_event, on='date_and_time', how='left')

sca_abs_dust_df['dust_event'] = sca_abs_dust_df['dust_event'].fillna(0).astype(int)

sca_abs_dust_df.sort_values(by='date_and_time', inplace=True)

sca_abs_dust_df.iloc[:, 1:] = sca_abs_dust_df.iloc[:, 1:].astype(float)

sca_abs_dust_df = sca_abs_dust_df.drop(columns=['sae_0', 'sae_1', 'sae_2'], errors='ignore')

sca_abs_dust_df = sca_abs_dust_df.merge(sae_0, on='date_and_time', how='left')
sca_abs_dust_df = sca_abs_dust_df.merge(sae_1, on='date_and_time', how='left')
sca_abs_dust_df = sca_abs_dust_df.merge(sae_2, on='date_and_time', how='left')

sca_abs_dust_df['date_and_time'] = pd.to_datetime(
    sca_abs_dust_df['date_and_time'], errors='coerce'
)

sca_abs_dust_df.iloc[:, 1:] = sca_abs_dust_df.iloc[:, 1:].astype(float)

start_date_range = pd.Timestamp('2010-10-04 00:00:00')
end_date_range = pd.Timestamp('2022-10-04 00:00:00')

sca_abs_dust_df = sca_abs_dust_df[
    (sca_abs_dust_df['date_and_time'] >= start_date_range) &
    (sca_abs_dust_df['date_and_time'] <= end_date_range)
]

df = pd.read_excel(r'./data\impactor\Impactor2004_2016.xlsx')

columns_to_keep = [
    'Small particle samplings with DEKATI PM-10 4-stage impactor in Hyytiälä 29.10.2004 - ', 
    'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14',
    'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18',
    'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22',
    'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26',
    'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30',
    'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33'
]

extracted_clean_impactor_data = df[columns_to_keep]

extracted_clean_impactor_data = extracted_clean_impactor_data.iloc[2:].reset_index(drop = True)

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(columns = ['Unnamed: 11'])

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {
    'Small particle samplings with DEKATI PM-10 4-stage impactor in Hyytiälä 29.10.2004 - ': 'start_date_and_time',
    'Unnamed: 12': 'end_date_and_time'
})

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {
    'Unnamed: 23': 'less_than_pm_1',
    'Unnamed: 24': 'between_1_and_2point5',
    'Unnamed: 25': 'between_2point5_and_10',
    'Unnamed: 26': 'more_than_pm_10',
    'Unnamed: 27': 'less_than_pm_10'
})

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {
    'Unnamed: 29': 'data_flag_for_1',
    'Unnamed: 30': 'data_flag_for_2',
    'Unnamed: 31': 'data_flag_for_3',
    'Unnamed: 32': 'data_flag_for_4'
})

extracted_clean_impactor_data[['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']] = extracted_clean_impactor_data[
    ['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']
].apply(pd.to_numeric, errors='coerce')

extracted_clean_impactor_data['start_date_and_time'] = pd.to_datetime(extracted_clean_impactor_data['start_date_and_time'], errors = 'coerce')
extracted_clean_impactor_data['end_date_and_time'] = pd.to_datetime(extracted_clean_impactor_data['end_date_and_time'], errors = 'coerce')

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {
    'Unnamed: 17': 'less_than_pm_1',
    'Unnamed: 18': 'between_1_and_2point5',
    'Unnamed: 19': 'between_2point5_and_10',
    'Unnamed: 20': 'more_than_pm_10',
    'Unnamed: 21': 'less_than_pm_10'
})

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {'Unnamed: 28': 'total_particles'})

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(columns = ['Unnamed: 16'])

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(columns = ['Unnamed: 13'])

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {'Unnamed: 14': 'time_duration_in_days'})

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(columns = ['Unnamed: 22'])

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {'Unnamed: 15': 'sampled_volume_of_air'})

extracted_clean_impactor_data = extracted_clean_impactor_data.iloc[1:, :]

extracted_clean_impactor_data.iloc[:, 2:-5] = extracted_clean_impactor_data.iloc[:, 2:-5].astype(float)

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(index = [1])

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(index = extracted_clean_impactor_data.index[2672:])

extracted_clean_impactor_data[['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']] = extracted_clean_impactor_data[
    ['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']
].fillna(0)

extracted_clean_impactor_data = extracted_clean_impactor_data.rename(columns = {'Unnamed: 33': 'notes'})

data_flags = [798, 797, 782, 781, 780, 771, 770, 760, 741, 740, 680, 679, 678, 676, 675, 674, 668, 665, 662, 660,
              657, 656, 655, 654, 653, 652, 651, 650, 649, 648, 647, 645, 644, 640, 632, 631, 630, 559, 558, 557,
              556, 555, 532, 531, 521, 499, 498, 476, 475, 472, 470, 458, 457, 450, 440, 420, 411, 410, 394, 392,
              390, 388, 382, 380, 370, 360, 299, 298, 276, 275, 258, 257, 250, 249, 248, 247, 220, 211, 210, 191,
              190, 189, 188, 187, 186, 185, 147, 120, 111, 110, 103, 102, 101, 100, 0, np.nan]

extracted_clean_impactor_data = extracted_clean_impactor_data[
    extracted_clean_impactor_data[['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']].isin(data_flags).any(axis = 1)
]

extracted_clean_impactor_data = extracted_clean_impactor_data[
    extracted_clean_impactor_data[['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']]
    .applymap(lambda x: x in data_flags)
    .any(axis = 1)
]

extracted_clean_impactor_data = extracted_clean_impactor_data[
    ~extracted_clean_impactor_data[['data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3', 'data_flag_for_4']]
    .isin([699])
    .any(axis = 1)
]

extracted_clean_impactor_data = extracted_clean_impactor_data.reset_index(drop = True)
extracted_clean_impactor_data['start_date_and_time'] = pd.to_datetime(extracted_clean_impactor_data['start_date_and_time'])
extracted_clean_impactor_data['end_date_and_time'] = pd.to_datetime(extracted_clean_impactor_data['end_date_and_time'])

extracted_clean_impactor_data = extracted_clean_impactor_data[
    (extracted_clean_impactor_data['start_date_and_time'] >= start_date_range) &
    (extracted_clean_impactor_data['end_date_and_time'] <= end_date_range)
]

extracted_clean_impactor_data.iloc[:, 2:] = extracted_clean_impactor_data.iloc[:, 2:].apply(
    lambda col: pd.to_numeric(col, errors='coerce')
)

mass_dates = []
averaged_optical_data = []

for i in range(len(extracted_clean_impactor_data)):
    start_time = extracted_clean_impactor_data.iloc[i]['start_date_and_time']
    end_time = extracted_clean_impactor_data.iloc[i]['end_date_and_time']

    mass_date = start_time + (end_time - start_time) / 2
    mass_dates.append(mass_date)
    
    filtered_data = sca_abs_dust_df[
        (sca_abs_dust_df['date_and_time'] >= start_time) & 
        (sca_abs_dust_df['date_and_time'] <= end_time)
    ]

    if not filtered_data.empty:
        optical_avg = filtered_data.drop(columns = ['date_and_time']).mean()
    else:
        optical_avg = pd.Series([None] * (sca_abs_dust_df.shape[1] - 1), index = sca_abs_dust_df.columns[1:])

    averaged_optical_data.append(optical_avg)

extracted_clean_impactor_data['mass_date_and_time'] = mass_dates

mass_date_column = extracted_clean_impactor_data.pop('mass_date_and_time')

new_column_names = [
    'start_date_and_time', 'end_date_and_time',
    'time_duration_in_days', 'sampled_volume_of_air', 'less_than_pm_1_x',
    'between_1_and_2point5_x', 'between_2point5_and_10_x',
    'more_than_pm_10_x', 'less_than_pm_10_x', 'less_than_pm_1_y',
    'between_1_and_2point5_y', 'between_2point5_and_10_y',
    'more_than_pm_10_y', 'less_than_pm_10_y', 'total_particles',
    'data_flag_for_1', 'data_flag_for_2', 'data_flag_for_3',
    'data_flag_for_4', 'notes'
]

extracted_clean_impactor_data.columns = new_column_names

columns_to_drop = [
    'sampled_volume_of_air',
    'less_than_pm_1_x', 
    'between_1_and_2point5_x', 
    'between_2point5_and_10_x', 
    'more_than_pm_10_x', 
    'less_than_pm_10_x',
    'total_particles'
]

extracted_clean_impactor_data = extracted_clean_impactor_data.drop(columns = columns_to_drop)

extracted_clean_impactor_data[['more_than_pm_10_y', 'less_than_pm_10_y']] = extracted_clean_impactor_data[['less_than_pm_10_y', 'more_than_pm_10_y']].values

extracted_clean_impactor_data.rename(columns = {'more_than_pm_10_y': 'temp_column_y', 'less_than_pm_10_y': 'more_than_pm_10_y'}, inplace = True)
extracted_clean_impactor_data.rename(columns = {'temp_column_y': 'less_than_pm_10_y'}, inplace = True)

extracted_clean_impactor_data[['data_flag_for_3', 'data_flag_for_4']] = extracted_clean_impactor_data[['data_flag_for_4', 'data_flag_for_3']].values

extracted_clean_impactor_data.rename(columns = {'data_flag_for_3': 'temp_column_z', 'data_flag_for_4': 'data_flag_for_3'}, inplace = True)
extracted_clean_impactor_data.rename(columns = {'temp_column_z': 'data_flag_for_4'}, inplace = True)

extracted_clean_impactor_data['between_1_and_10'] = extracted_clean_impactor_data['less_than_pm_10_y'] - extracted_clean_impactor_data['less_than_pm_1_y']

extracted_clean_impactor_data.insert(2, 'mass_date_and_time', mass_date_column)

extracted_clean_impactor_data.iloc[:, 0] = pd.to_datetime(extracted_clean_impactor_data.iloc[:, 0])

extracted_clean_impactor_data.iloc[:, 1] = pd.to_datetime(extracted_clean_impactor_data.iloc[:, 1])

extracted_clean_impactor_data.iloc[:, 2] = pd.to_datetime(extracted_clean_impactor_data.iloc[:, 2])

for col in extracted_clean_impactor_data.columns[3:]:
    extracted_clean_impactor_data[col] = extracted_clean_impactor_data[col].apply(float)
    
extracted_clean_impactor_data = extracted_clean_impactor_data.reset_index(drop = True)

extracted_clean_impactor_data['start_date_and_time'] = pd.to_datetime(extracted_clean_impactor_data['start_date_and_time'])
extracted_clean_impactor_data['end_date_and_time'] = pd.to_datetime(extracted_clean_impactor_data['end_date_and_time'])

sca_abs_dust_df['date_and_time'] = pd.to_datetime(sca_abs_dust_df['date_and_time'])

optical_and_mass_data = pd.DataFrame()

for idx, row in extracted_clean_impactor_data.iterrows():
    start_time = row['start_date_and_time']
    end_time = row['end_date_and_time']

    midpoint_time = start_time + (end_time - start_time) / 2
 
    mask = (sca_abs_dust_df['date_and_time'] >= start_time) & (sca_abs_dust_df['date_and_time'] <= end_time)
    filtered_data = sca_abs_dust_df[mask]

    if not filtered_data.empty:
        averages = filtered_data.mean()
    else:
        averages = pd.Series(index=sca_abs_dust_df.columns)

    for col in extracted_clean_impactor_data.columns:
        averages[col] = row[col]

    averages['optical_date_and_time'] = midpoint_time

    optical_and_mass_data = optical_and_mass_data.append(averages, ignore_index = True)

column_order = ['optical_date_and_time'] + [col for col in extracted_clean_impactor_data.columns if col != 'optical_date_and_time']
optical_and_mass_data = optical_and_mass_data[column_order + [col for col in optical_and_mass_data.columns if col not in column_order]]

optical_and_mass_data.reset_index(drop = True, inplace = True)

dust_event['date_and_time'] = pd.to_datetime(dust_event['date_and_time'])
optical_and_mass_data['start_date_and_time'] = pd.to_datetime(optical_and_mass_data['start_date_and_time'])
optical_and_mass_data['end_date_and_time'] = pd.to_datetime(optical_and_mass_data['end_date_and_time'])

optical_and_mass_data['date_and_time'] = pd.NaT
optical_and_mass_data['dust_event'] = np.nan

def find_dust_events(start, end):
    mask = (dust_event['date_and_time'] >= start) & (dust_event['date_and_time'] <= end)
    events = dust_event[mask]
    if not events.empty:
        return events.iloc[-1]['date_and_time'], events.iloc[-1]['dust_event']
    return pd.NaT, np.nan

for idx, row in optical_and_mass_data.iterrows():
    date, event = find_dust_events(row['start_date_and_time'], row['end_date_and_time'])
    optical_and_mass_data.at[idx, 'date_and_time'] = date
    optical_and_mass_data.at[idx, 'dust_event'] = event

cols = list(optical_and_mass_data.columns)
cols = [col for col in cols if col not in ['date_and_time', 'dust_event']] + ['date_and_time', 'dust_event']
optical_and_mass_data = optical_and_mass_data[cols]

exclude_cols = [
    'optical_date_and_time', 'dust_event',
    'aae_0', 'aae_1', 'aae_2',
    'sae_0', 'sae_1', 'sae_2'
]

optical_and_mass_data = optical_and_mass_data.rename(columns={
    col: col + '_custom_mean' for col in sca_abs_dust_df.columns
    if isinstance(col, str) and col not in exclude_cols
})

mask_pm10 = optical_and_mass_data['less_than_pm_10_y'] != 0
mask_pm1 = optical_and_mass_data['less_than_pm_1_y'] != 0
mask_10_1 = optical_and_mass_data['between_1_and_10'] != 0

optical_and_mass_data['mac_pm_10_520_custom_mean'] = np.nan
optical_and_mass_data['mac_pm_1_520_custom_mean'] = np.nan
optical_and_mass_data['mac_10_1_520_custom_mean'] = np.nan
optical_and_mass_data['msc_pm_10_550_custom_mean'] = np.nan
optical_and_mass_data['msc_pm_1_550_custom_mean'] = np.nan
optical_and_mass_data['msc_10_1_550_custom_mean'] = np.nan

optical_and_mass_data.loc[mask_pm10, 'mac_pm_10_520_custom_mean'] = (
    optical_and_mass_data.loc[mask_pm10, 'pm_10_abs_520_custom_mean'] /
    optical_and_mass_data.loc[mask_pm10, 'less_than_pm_10_y']
)

optical_and_mass_data.loc[mask_pm1, 'mac_pm_1_520_custom_mean'] = (
    optical_and_mass_data.loc[mask_pm1, 'pm_1_abs_520_custom_mean'] /
    optical_and_mass_data.loc[mask_pm1, 'less_than_pm_1_y']
)

optical_and_mass_data.loc[mask_10_1, 'mac_10_1_520_custom_mean'] = (
    optical_and_mass_data.loc[mask_10_1, 'abs_10_1_520_custom_mean'] /
    optical_and_mass_data.loc[mask_10_1, 'between_1_and_10']
)

optical_and_mass_data.loc[mask_pm10, 'msc_pm_10_550_custom_mean'] = (
    optical_and_mass_data.loc[mask_pm10, 'pm_10_sca_550_custom_mean'] /
    optical_and_mass_data.loc[mask_pm10, 'less_than_pm_10_y']
)

optical_and_mass_data.loc[mask_pm1, 'msc_pm_1_550_custom_mean'] = (
    optical_and_mass_data.loc[mask_pm1, 'pm_1_sca_550_custom_mean'] /
    optical_and_mass_data.loc[mask_pm1, 'less_than_pm_1_y']
)

optical_and_mass_data.loc[mask_10_1, 'msc_10_1_550_custom_mean'] = (
    optical_and_mass_data.loc[mask_10_1, 'sca_10_1_550_custom_mean'] /
    optical_and_mass_data.loc[mask_10_1, 'between_1_and_10']
)

optical_and_mass_data = optical_and_mass_data.replace({pd.NaT: np.nan})

optical_and_mass_data.columns = optical_and_mass_data.columns.str.replace('_custom_mean$', '', regex=True)

optical_and_mass_data.set_index('optical_date_and_time', inplace=True)

start_date = pd.to_datetime('2010-10-04 00:00:00')
end_date = pd.to_datetime('2022-10-04 00:00:00')

optical_and_mass_data = optical_and_mass_data.loc[(optical_and_mass_data.index >= start_date) & (optical_and_mass_data.index <= end_date)]

sns.set_style('whitegrid')
plt.close('all')
plt.rcdefaults()

POLLEN_CODE    = 555
DUST_TRUE      = {1}
USE_BACKGROUND = False

OUT_TS_PM10_ANN   = "ts_pm10_annotated.png"
OUT_TS_PM10_CLEAN = "ts_pm10.png"
OUT_TS_SPM_ANN    = "ts_spm10_annotated.png"
OUT_TS_SPM_CLEAN  = "ts_spm10.png"
OUT_BARS_MASS     = "figure_3_seasonal_mass_concentration_COMMON.png"
OUT_BARS_PCT      = "figure_4_seasonal_mass_fraction_COMMON.png"        

MASS_YLABEL = r"PM mass ($\mu$gm$^{-3}$)"
SLOPE_UNIT  = r"$\mu$gm$^{-3}$yr$^{-1}$"
REL_UNIT    = r"%yr$^{-1}$"

YLIM_PM10 = (0, 25)
YLIM_SPM  = (0, 5)

plt.rcParams.update({
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
    "axes.unicode_minus": True,
    "font.size": 60,
    "axes.labelsize": 60,
    "xtick.labelsize": 60,
    "ytick.labelsize": 60,
    "axes.linewidth": 3,
    "axes.spines.top": True,
    "axes.spines.right": True,
})
CORN = "cornflowerblue"

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def modified_z_score(a, threshold=3.5):
    a = np.asarray(a, dtype=float)
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    if not np.isfinite(mad) or mad == 0:
        return np.full(a.shape, False, dtype=bool)
    z = 0.6745 * (a - med) / mad
    return np.abs(z) > threshold

def custom_round(value: float) -> float:
    s = f"{float(value):.3f}"
    last = int(s[-1])
    out = Decimal(s).quantize(Decimal("1.00"), rounding=ROUND_HALF_UP)
    if last >= 5:
        out += Decimal("0.01")
    return float(out)

def generate_ticks(lo, hi, n=5):
    loc = MaxNLocator(nbins=n, prune=None)
    ticks = [t for t in loc.tick_values(lo, hi) if lo <= t <= hi]
    return [custom_round(t) for t in ticks]

def fmt_y(v, _):
    return f"{v:.2f}"

def month_labels():
    return [f"{m:02d}" for m in range(1, 13)]

def format_sci_x10(v, decimals=2, plain_bounds=(1e-2, 1e2)):
    if v is None or not np.isfinite(v):
        return "NaN"
    v = float(v)
    if v == 0 or (plain_bounds[0] <= abs(v) < plain_bounds[1]):
        q = Decimal(str(v)).quantize(Decimal(f"1.{'0'*decimals}"), rounding=ROUND_HALF_UP)
        return f"{q:.{decimals}f}"
    exp = int(np.floor(np.log10(abs(v))))
    coeff = v / (10 ** exp)
    return rf"{coeff:.{decimals}f} x 10$^{{{exp}}}$"

EXCEPTIONS_3DP = {Decimal('0.051'), Decimal('0.052'), Decimal('0.053'), Decimal('0.054')}

def p_round(p, places):
    return Decimal(str(p)).quantize(Decimal('0.' + '0'*places), rounding=ROUND_HALF_UP)

def normalize_p(p):
    """Return (p rounded to 3 d.p. as Decimal, significance by that 3-d.p. value)."""
    if not np.isfinite(p):
        return Decimal('NaN'), False
    p3 = p_round(p, 3)
    return p3, float(p3) <= 0.050

def format_scientific_pow10_custom(v):
    if not np.isfinite(v) or v == 0:
        return "0.00"
    a = abs(v)
    exp = int(np.floor(np.log10(a)))
    mant = a / (10 ** exp)

    mant3 = Decimal(str(mant)).quantize(Decimal('1.000'), rounding=ROUND_HALF_UP)
    s = f"{mant3:.3f}"
    w, xy = s.split('.')
    x, y, z = int(xy[0]), int(xy[1]), int(xy[2])

    if z <= 4:
        mant2 = float(f"{w}.{x}{y}")
    else:
        y_up = y + 1
        if y_up < 10:
            mant2 = float(f"{w}.{x}{y_up}")
        else:
            x_up = x + 1
            if x_up < 10:
                mant2 = float(f"{w}.{x_up}0")
            else:
                mant2 = 1.00
                exp += 1
                return f"{mant2:.2f} x 10$^{{{exp}}}$"

    if mant2 >= 10.0:
        mant2 = 1.00
        exp += 1

    return f"{mant2:.2f} x 10$^{{{exp}}}$"

def format_p_display(p):
    if not np.isfinite(p):
        return "NaN"
    p3, _ = normalize_p(p)
    if float(p) < 0.01:
        return format_scientific_pow10_custom(p)
    if p3 in EXCEPTIONS_3DP:
        return f"{float(p3):.3f}"
    return f"{float(p_round(p, 2)):.2f}"

def effective_sample_size(series, nlags=None):
    x = pd.Series(series).dropna()
    n = len(x)
    if n < 3:
        return 1, n
    if nlags is None:
        nlags = max(1, min(int(n/4), 20))
    acfs_vals = acf(x, nlags=nlags, fft=True)[1:]
    signif = 1.96 / np.sqrt(n)
    acfs_sig = acfs_vals[np.abs(acfs_vals) > signif]
    ess = n / (1 + 2 * np.sum(acfs_sig))
    return max(ess, 1), n

def modified_mann_kendall_trend(x, y, nlags=None):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    ess, _ = effective_sample_size(y, nlags=nlags)
    tau, _ = kendalltau(x, y)
    z = tau * np.sqrt(9 * ess * (ess - 1) / (2 * (2 * ess + 5)))
    p = 2 * (1 - norm.cdf(abs(z)))
    return tau, p

def build_event_masks(df: pd.DataFrame):
    f1 = _num(df.get('data_flag_for_1', np.nan)).eq(POLLEN_CODE)
    f2 = _num(df.get('data_flag_for_2', np.nan)).eq(POLLEN_CODE)
    f3 = _num(df.get('data_flag_for_3', np.nan)).eq(POLLEN_CODE)
    f4 = _num(df.get('data_flag_for_4', np.nan)).eq(POLLEN_CODE)
    pol_pm10 = (f1 | f2 | f3).reindex(df.index, fill_value=False).to_numpy()
    pol_spm  =  f4.reindex(df.index, fill_value=False).to_numpy()
    dust = _num(df.get('dust_event', np.nan)).isin(DUST_TRUE).reindex(df.index, fill_value=False).to_numpy()
    return pol_pm10, pol_spm, dust

def theilsen_and_p(x_dt, y_vals):
    x_dt = pd.to_datetime(x_dt)
    y = pd.to_numeric(pd.Series(y_vals), errors="coerce").to_numpy()
    y[modified_z_score(y)] = np.nan
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([])
    x_clean = x_dt[m]
    y_clean = y[m]
    t_years = ((x_clean - x_clean.min()) / np.timedelta64(1, 'D')).astype(float) / 365.25
    slope, intercept, lo, hi = theilslopes(y_clean, t_years, alpha=0.95)
    unc = abs(hi - lo) / 2.0
    _, p = modified_mann_kendall_trend(t_years, y_clean)
    return slope, intercept, unc, p, t_years, y_clean

def style_ticks(ax):
    ax.minorticks_off()
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x', which='major', direction='out', colors='black', width=6, length=18)
    ax.tick_params(axis='y', which='major', direction='out', colors='black', width=6, length=18)

def plot_trend_panel(df, y_series, pollen_mask, dust_mask, outfile, ylabel, ylim=None, annotate=True):
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.set_facecolor("white")
    ax.set_axisbelow(True)

    x_all = pd.to_datetime(df.index).to_numpy()
    slope, intercept, unc, p, t_years, y_clean = theilsen_and_p(x_all, y_series)

    ax.scatter(df.index, y_series, s=300, facecolors='none',
               edgecolors=CORN, linewidth=3, zorder=5)

    if ylim is None:
        ymax_val = float(np.nanmax(y_series)) if np.isfinite(np.nanmax(y_series)) else 1.0
        ymin, ymax = 0.0, (ymax_val * 1.10 if ymax_val > 0 else 1.0)
    else:
        ymin, ymax = ylim
    ax.set_ylim((ymin, ymax))

    y_star_pollen = np.where(np.isfinite(y_series), y_series, ymax * 0.96)
    y_star_dust   = np.where(np.isfinite(y_series), y_series, ymax * 0.92)
    ax.scatter(df.index[pollen_mask], y_star_pollen[pollen_mask],
               marker='*', s=600, color='green', edgecolors='green', linewidth=3, zorder=7)
    ax.scatter(df.index[dust_mask],   y_star_dust[dust_mask],
               marker='*', s=600, color='red',   edgecolors='red',   linewidth=3, zorder=7)

    _, is_sig = normalize_p(p)
    if np.isfinite(slope):
        x0 = pd.to_datetime(df.index.min())
        x1 = pd.to_datetime(df.index.max())
        t0 = 0.0
        t1 = ((x1 - x0) / np.timedelta64(1, 'D')) / 365.25
        y0 = intercept + slope * t0
        y1 = intercept + slope * t1
        ax.plot([x0, x1], [y0, y1], color='red', linewidth=6,
                linestyle='-' if is_sig else '--', zorder=8)

    ax.set_xlabel('Year', fontsize=60, labelpad=30)
    ax.set_ylabel(ylabel, fontsize=60, labelpad=30)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax.set_yticks(generate_ticks(*ax.get_ylim(), n=6))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    style_ticks(ax)

    for s in ax.spines.values():
        s.set_color('black'); s.set_linewidth(6)
    ax.grid(True, linestyle='--', color='gray', linewidth=3)

    if annotate and np.isfinite(slope) and t_years.size:
        med = float(np.nanmedian(y_clean))
        rel = (slope / abs(med)) * 100.0 if (np.isfinite(med) and med != 0) else np.nan
        rel_unc = (unc / abs(med)) * 100.0 if (np.isfinite(med) and med != 0) else np.nan

        p_label = "*p-value" if is_sig else "p-value"
        p_txt   = f"{p_label}: {format_p_display(p)}"
        slope_txt = rf"Slope: {format_sci_x10(slope, 2)} ± {format_sci_x10(unc, 2)} {SLOPE_UNIT}"
        rel_txt   = rf"Relative trend: {format_sci_x10(rel, 2)} ± {format_sci_x10(rel_unc, 2)} {REL_UNIT}"

        ax.text(0.02, 0.96, p_txt,     transform=ax.transAxes, fontsize=60, va='top')
        ax.text(0.02, 0.81, slope_txt, transform=ax.transAxes, fontsize=60, va='top')
        ax.text(0.02, 0.66, rel_txt,   transform=ax.transAxes, fontsize=60, va='top')

    plt.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.20)
    plt.savefig(outfile, dpi=300)
    plt.show()

def compute_daily_common(df: pd.DataFrame, use_background: bool):
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)

    pol_pm10, pol_spm, dust = build_event_masks(d)
    if use_background:
        ok = ~(dust | pol_pm10 | pol_spm)
        d = d.loc[ok]

    pm10_d = (_num(d['less_than_pm_1_y']) +
              _num(d['between_1_and_2point5_y']) +
              _num(d['between_2point5_and_10_y'])).resample('D').mean()
    spm_d  = _num(d['more_than_pm_10_y']).resample('D').mean()

    common = pm10_d.notna() & spm_d.notna()
    pm10_d, spm_d = pm10_d[common], spm_d[common]
    with np.errstate(divide='ignore', invalid='ignore'):
        f_day = spm_d / (pm10_d + spm_d)
    return pm10_d, spm_d, f_day

def monthly_from_daily(pm10_d, spm_d, f_day):
    pm10_m = pm10_d.resample('MS').mean().groupby(lambda d: d.month).mean()
    spm_m  = spm_d .resample('MS').mean().groupby(lambda d: d.month).mean()
    f_m    = f_day.resample('MS').mean().groupby(lambda d: d.month).mean()
    idx = pd.Index(range(1, 13), name="Month")
    return pm10_m.reindex(idx), spm_m.reindex(idx), f_m.reindex(idx)

def plot_monthly_mass(pm10_m, spm_m, outfile):
    fig, ax = plt.subplots(figsize=(30, 10))
    bar_w = 0.38
    x = np.arange(1, 13)

    ax.bar(x - bar_w/2, pm10_m.values, bar_w, color='cornflowerblue', zorder=3)
    ax.bar(x + bar_w/2, spm_m.values,  bar_w, color='red',            zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(month_labels())
    ax.set_xlabel('Month', fontsize=60, labelpad=10)
    ax.set_ylabel(MASS_YLABEL, fontsize=60, labelpad=30)
    ax.set_ylim(0.00, 10.00)
    ax.set_yticks([0.00, 2.00, 4.00, 6.00, 8.00, 10.00])
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))
    style_ticks(ax)

    for s in ax.spines.values(): s.set_color('black'); s.set_linewidth(6)
    ax.set_axisbelow(False)
    ax.yaxis.grid(True, linestyle='--', color='gray', linewidth=3, alpha=0.9, zorder=6)
    ax.xaxis.grid(True, linestyle='--', color='gray', linewidth=3, alpha=0.9, zorder=0)

    plt.tight_layout(); plt.savefig(outfile, dpi=300, bbox_inches='tight'); plt.show()

def plot_monthly_percent(f_m, outfile):
    fig, ax = plt.subplots(figsize=(30, 10))
    bar_w = 0.38
    x = np.arange(1, 13)

    pm10_pct = 100.0 * (1.0 - f_m.values)
    spm_pct  = 100.0 * f_m.values

    ax.bar(x - bar_w/2, pm10_pct, bar_w, color='cornflowerblue', zorder=3)
    ax.bar(x + bar_w/2, spm_pct,  bar_w, color='red',            zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(month_labels())
    ax.set_xlabel('Month', fontsize=60, labelpad=10)
    ax.set_ylabel('PM mass (%)', fontsize=60, labelpad=30)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    style_ticks(ax)

    for s in ax.spines.values(): s.set_color('black'); s.set_linewidth(6)
    ax.set_axisbelow(False)
    ax.yaxis.grid(True, linestyle='--', color='gray', linewidth=3, alpha=0.9, zorder=6)
    ax.xaxis.grid(True, linestyle='--', color='gray', linewidth=3, alpha=0.9, zorder=0)

    plt.tight_layout(); plt.savefig(outfile, dpi=300, bbox_inches='tight'); plt.show()

if __name__ == "__main__":
    try:
        optical_and_mass_data
    except NameError:
        raise RuntimeError("Define `optical_and_mass_data` (DataFrame with a DatetimeIndex) before running.")

    df = optical_and_mass_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    pol_pm10, pol_spm, dust = build_event_masks(df)

    pm10_series = (_num(df['less_than_pm_1_y']) +
                   _num(df['between_1_and_2point5_y']) +
                   _num(df['between_2point5_and_10_y'])).to_numpy()
    plot_trend_panel(df, pm10_series, pol_pm10, dust, OUT_TS_PM10_ANN,  MASS_YLABEL, YLIM_PM10, True)
    plot_trend_panel(df, pm10_series, pol_pm10, dust, OUT_TS_PM10_CLEAN, MASS_YLABEL, YLIM_PM10, False)

    spm_series = _num(df['more_than_pm_10_y']).to_numpy()
    plot_trend_panel(df, spm_series, pol_spm,  dust, OUT_TS_SPM_ANN,   MASS_YLABEL, YLIM_SPM, True)
    plot_trend_panel(df, spm_series, pol_spm,  dust, OUT_TS_SPM_CLEAN, MASS_YLABEL, YLIM_SPM, False)

    pm10_d, spm_d, f_day = compute_daily_common(df, use_background=USE_BACKGROUND)
    pm10_m, spm_m, f_m   = monthly_from_daily(pm10_d, spm_d, f_day)
    plot_monthly_mass(pm10_m, spm_m, OUT_BARS_MASS)
    plot_monthly_percent(f_m, OUT_BARS_PCT)

OUT_DIR  = os.getcwd()
FIGSIZE  = (15, 5)

DEFAULT_FONT_SIZE  = 30
FONT_SIZE_COUNTS   = 30
FONT_SIZE_PERCENT  = 30  

SPINE_W  = 3
TICK_W   = 3
TICKLEN  = 9

GRID_WY  = 2                      
GRID_WX  = 2                      
GRID_Y_ALPHA = 1
GRID_X_ALPHA = 1
GRAY = (0.5, 0.5, 0.5)              


Y_DASH = (6.0, 1.0)                 
X_DASH = (6.0, 1.0)                 
HALO_W = 0.8

POLLEN_CODE = 555                   
DUST_TRUE   = {1, True}

MANUAL_HINTS = {
    "pollen_flag_cols": [
        "data_flag_for_1", "data_flag_for_2", "data_flag_for_3", "data_flag_for_4",
    ],
    "dust_flag_cols": [
        "dust_event",
    ],
}

os.makedirs(OUT_DIR, exist_ok=True)

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def _cols_matching(df, *patterns):
    out = []
    for c in df.columns:
        name = str(c)
        if all(re.search(p, name, flags=re.I) for p in patterns):
            out.append(c)
    return out

def _unique_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def detect_size_fraction_cols(df):
    pollen_candidates = []
    dust_candidates   = []
    size_patterns = [r"pm\s*1(?!0)", r"pm\s*10", r"pm\s*1\s*[-_]?\s*10", r"(spm|super)"]

    for size_pat in size_patterns:
        pollen_candidates += _cols_matching(df, r"(data_)?flag|pollen", size_pat)
        pollen_candidates += _cols_matching(df, r"data_flag", size_pat)

    for size_pat in size_patterns:
        dust_candidates   += _cols_matching(df, r"dust", size_pat)

    for c in MANUAL_HINTS["pollen_flag_cols"]:
        if c in df.columns:
            pollen_candidates.append(c)
    for c in MANUAL_HINTS["dust_flag_cols"]:
        if c in df.columns:
            dust_candidates.append(c)

    if not dust_candidates and "dust_event" in df.columns:
        dust_candidates = ["dust_event"]
    if not pollen_candidates:
        pollen_candidates = [c for c in df.columns if re.search(r"data.*flag", str(c), re.I)]

    return {
        "pollen_cols": _unique_keep_order(pollen_candidates),
        "dust_cols":   _unique_keep_order(dust_candidates),
    }

def build_event_masks(df: pd.DataFrame):
    found = detect_size_fraction_cols(df)
    pol_cols, du_cols = found["pollen_cols"], found["dust_cols"]

    pol_any_series = None
    for c in pol_cols:
        s = _num(df.get(c, np.nan)).eq(POLLEN_CODE)
        pol_any_series = s if pol_any_series is None else (pol_any_series | s)
    pollen_any = (pol_any_series.reindex(df.index, fill_value=False).to_numpy()
                  if pol_any_series is not None else np.zeros(len(df), dtype=bool))

    dust_any_series = None
    for c in du_cols:
        s = _num(df.get(c, np.nan)).isin(DUST_TRUE)
        dust_any_series = s if dust_any_series is None else (dust_any_series | s)
    dust_any = (dust_any_series.reindex(df.index, fill_value=False).to_numpy()
                if dust_any_series is not None else np.zeros(len(df), dtype=bool))

    print("Detected pollen columns:", pol_cols or "None")
    print("Detected dust columns:",   du_cols  or "None")
    return pollen_any, dust_any

def monthly_event_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        if "optical_date_and_time" in d:
            d.index = pd.to_datetime(d["optical_date_and_time"])
        else:
            d.index = pd.to_datetime(d.index)

    pollen_any, dust_any = build_event_masks(d)

    pollen_day = pd.Series(pollen_any, index=d.index).resample("D").max().fillna(False)
    dust_day   = pd.Series(dust_any,   index=d.index).resample("D").max().fillna(False)

    observed   = pd.Series(True, index=d.index).resample("D").max().fillna(False)

    month_idx = np.arange(1, 13)
    def msum(s: pd.Series) -> pd.Series:
        return (s.groupby(s.index.month).sum().reindex(month_idx, fill_value=0).astype("float64"))

    pollen_counts = msum(pollen_day)
    dust_counts   = msum(dust_day)
    denom_days    = msum(observed)

    with np.errstate(invalid="ignore", divide="ignore"):
        pollen_pct = 100.0 * (pollen_counts / denom_days.replace(0.0, np.nan))
        dust_pct   = 100.0 * (dust_counts   / denom_days.replace(0.0, np.nan))

    out = pd.DataFrame({
        "pollen_days":   pollen_counts.astype("int64"),
        "dust_days":     dust_counts.astype("int64"),
        "observed_days": denom_days.astype("int64"),
        "pollen_pct":    pollen_pct.astype("float64").round(2),
        "dust_pct":      dust_pct.astype("float64").round(2),
    }, index=month_idx)
    out.index.name = "month"
    return out

def month_labels_double_digits():
    return [f"{m:02d}" for m in range(1, 13)]

def _nice_upper(ymax):
    if not np.isfinite(ymax) or ymax <= 0:
        return 1
    step = 1 if ymax <= 10 else 2 if ymax <= 20 else 5
    return (int(np.ceil(ymax / step)) * step) * 1.25

def _save(fig, filename):
    path = os.path.abspath(os.path.join(OUT_DIR, filename))
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    print("Saved:", path)
    return path

def list_event_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        if "optical_date_and_time" in d:
            d.index = pd.to_datetime(d["optical_date_and_time"])
        else:
            d.index = pd.to_datetime(d.index)

    pollen_any, dust_any = build_event_masks(d)
    s_pol = pd.Series(pollen_any, index=d.index)
    s_dus = pd.Series(dust_any,   index=d.index)

    def first_times(s_bool: pd.Series, label: str) -> pd.DataFrame:
        if not s_bool.any():
            return pd.DataFrame(columns=["event_type","date","first_datetime_that_day"])
        subset = s_bool[s_bool]
        ft = subset.groupby(subset.index.normalize()).apply(lambda x: x.index.min())
        return pd.DataFrame({
            "event_type": label,
            "date": ft.index.date,
            "first_datetime_that_day": ft.values
        })

    df_pol = first_times(s_pol, "pollen")
    df_dus = first_times(s_dus, "dust")
    df_events = (pd.concat([df_pol, df_dus], ignore_index=True)
                   .sort_values("first_datetime_that_day")
                   .reset_index(drop=True))
    return df_events

def plot_grouped_bars(
    y1, y2, ylabel, title, outfile, *,
    integer_y=False, annotate=True, show=True, font_size=DEFAULT_FONT_SIZE
):
    
    def _clean(y):
        a = np.asarray(y, dtype=float)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    y1 = _clean(y1); y2 = _clean(y2)

    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "patch.edgecolor": "none",
        "patch.linewidth": 0.0,
        "patch.antialiased": False,
        "lines.dash_capstyle": "butt",
    }):
        plt.close('all')
        x = np.arange(1, 13)
        w = 0.42
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)

        bars1 = ax.bar(x + w/2, y1, width=w, zorder=3, label="Pollen", color="green", edgecolor='none', linewidth=0)
        bars2 = ax.bar(x - w/2, y2, width=w, zorder=3, label="Dust", color="red", edgecolor='none', linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels(month_labels_double_digits())
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.set_axisbelow(True)

        ax.grid(True, axis="y",
                linestyle=(0, Y_DASH),
                linewidth=GRID_WY,
                color=(*GRAY, GRID_Y_ALPHA),
                zorder=0)
        for gl in ax.get_ygridlines():
            gl.set_dash_capstyle('butt')

        ax.grid(False, axis="x")
        for xi in x:
            vl = ax.axvline(xi,
                            linestyle=(0, X_DASH),
                            linewidth=GRID_WX,
                            color=(*GRAY, GRID_X_ALPHA),
                            zorder=0)
            vl.set_dash_capstyle('butt')

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(SPINE_W)
            spine.set_color("black")

        ax.tick_params(axis="both", direction="out", width=TICK_W, length=TICKLEN, colors="black")
        ax.margins(x=0.02, y=0.08)

        ymax = float(max(y1.max(initial=0.0), y2.max(initial=0.0)))
        ax.set_ylim(0, _nice_upper(ymax))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=integer_y))

        if annotate:
            for b in [*bars1, *bars2]:
                v = b.get_height()
                if v > 0:
                    txt = f"{int(round(v))}" if integer_y else f"{v:.0f}%"
                    ax.text(b.get_x() + b.get_width()/2, v + 0.6, txt,
                            ha="center", va="bottom", fontsize=font_size, clip_on=False,
                            path_effects=[pe.withStroke(linewidth=HALO_W, foreground="white")])

        plt.tight_layout(pad=0.8)
        _save(fig, outfile)
        if show:
            plt.show()
        else:
            plt.close(fig)

if 'optical_and_mass_data' not in globals():
    raise RuntimeError("`optical_and_mass_data` must exist as a pandas DataFrame.")

stats_m = monthly_event_stats(optical_and_mass_data)

total_pollen = int(stats_m['pollen_days'].sum())
total_dust   = int(stats_m['dust_days'].sum())
total_events = total_pollen + total_dust
print(f"Total events (pollen + dust) = {total_events}")

ylabel_counts = rf'$\mathit{{n}}_{{\mathit{{total}}}} \;=\; {total_events}$'

plot_grouped_bars(
    stats_m['pollen_days'].values,
    stats_m['dust_days'].values,
    ylabel=ylabel_counts,
    title='',
    outfile='monthly_event_counts.png',
    integer_y=True, annotate=True, show=True,
    font_size=FONT_SIZE_COUNTS
)

plot_grouped_bars(
    stats_m['pollen_pct'].values,
    stats_m['dust_pct'].values,
    ylabel='%',
    title='',               
    outfile='monthly_event_frequency_percent.png',
    integer_y=False, annotate=True, show=True,     
    font_size=FONT_SIZE_PERCENT
)

stats_csv_path = os.path.abspath(os.path.join(OUT_DIR, 'monthly_pollen_dust_stats.csv'))
stats_m.to_csv(stats_csv_path, index_label='month')
print("Saved:", stats_csv_path)

events_df = list_event_datetimes(optical_and_mass_data)

if len(events_df) != total_events:
    print(f"[WARN] Event rows ({len(events_df)}) != pollen+dust total ({total_events}). "
          "This can happen if your daily aggregation or columns differ from assumptions.")

print("\nEvent datetimes (first hit per day for each phenomenon):")
if len(events_df) > 0:
    print(events_df.to_string(index=False))
else:
    print("(no events found)")

events_csv_path = os.path.abspath(os.path.join(OUT_DIR, 'pollen_dust_event_datetimes.csv'))
events_df.to_csv(events_csv_path, index=False)
print("Saved:", events_csv_path)    

LABEL_FONT_PT   = 13.25
LABEL_WEIGHT    = "normal"

PATCH_W_PT      = 10.25
PATCH_H_PT      = 28

GAP_SYM_TXT_PT  = 4           
ENTRY_GAP_PT    = 374         
MARGIN_L_PT     = 0.25        
MARGIN_R_PT     = 0.25        
MARGIN_T_PT     = 0
MARGIN_B_PT     = 0

DPI             = 300
HEIGHT_SCALE    = 0.85
OUT_PNG         = "legend_dust_pollen_trimmed_edges_thinner_again.png"
OUT_SVG         = "legend_dust_pollen_trimmed_edges_thinner_again.svg"

DUST_COLOR      = "red"
POLLEN_COLOR    = "green"

def text_size_px(s, fontsize, dpi, weight=None):
    fig = plt.figure(figsize=(1, 1), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    t = ax.text(0, 0, s, fontsize=fontsize, weight=weight)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)
    return bb.width, bb.height

def pt_to_px(pt, dpi=DPI):
    return (pt / 72.0) * dpi

w_dust_px,   h_dust_px   = text_size_px("Dust",   LABEL_FONT_PT, DPI, LABEL_WEIGHT)
w_pollen_px, h_pollen_px = text_size_px("Pollen", LABEL_FONT_PT, DPI, LABEL_WEIGHT)

patch_w  = pt_to_px(PATCH_W_PT)
patch_h  = pt_to_px(PATCH_H_PT)
gap_st   = pt_to_px(GAP_SYM_TXT_PT)
gap_ent  = pt_to_px(ENTRY_GAP_PT)
mL, mR   = pt_to_px(MARGIN_L_PT), pt_to_px(MARGIN_R_PT)
mT, mB   = pt_to_px(MARGIN_T_PT), pt_to_px(MARGIN_B_PT)

entry1_w  = patch_w + gap_st + w_dust_px
entry2_w  = patch_w + gap_st + w_pollen_px
content_w = entry1_w + gap_ent + entry2_w
content_h = max(patch_h, h_dust_px, h_pollen_px)

base_w_in = (content_w + mL + mR) / DPI
base_h_in = (content_h + mT + mB) / DPI
fig_w_in  = base_w_in
fig_h_in  = base_h_in * HEIGHT_SCALE

fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=DPI)
ax  = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, content_w + mL + mR)
ax.set_ylim(0, content_h + mB + mT)

y_center = mB + content_h / 2.0
x = mL

def vpatch_square(ax, x_left, y_mid, w, h, color):
    return ax.add_patch(Rectangle((x_left, y_mid - h/2), w, h,
                                  linewidth=0, facecolor=color, edgecolor='none'))

vpatch_square(ax, x, y_center, patch_w, patch_h, DUST_COLOR)
x_text = x + patch_w + gap_st
ax.text(x_text, y_center, "Dust",
        fontsize=LABEL_FONT_PT, weight=LABEL_WEIGHT, va='center', ha='left')

x = x_text + w_dust_px + gap_ent
vpatch_square(ax, x, y_center, patch_w, patch_h, POLLEN_COLOR)
x_text = x + patch_w + gap_st
ax.text(x_text, y_center, "Pollen",
        fontsize=LABEL_FONT_PT, weight=LABEL_WEIGHT, va='center', ha='left')

fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.0, facecolor="none")
fig.savefig(OUT_SVG,            bbox_inches="tight", pad_inches=0.0, facecolor="none")
print(f"Saved: {OUT_PNG} and {OUT_SVG}")
plt.show()