import datetime
import pandas as pd
import numpy as np
from typing import Tuple
import re

# def serialize_data(ser: pd.Series, padding=0) -> Tuple[np.ndarray, np.ndarray]:
#     mask = pd.notna(ser).astype(int)
#     r = pd.Series([np.nan] * padding).append(ser).append(pd.Series([np.nan] * padding)).to_numpy()
#     mask = np.concatenate((np.zeros(padding), mask, np.zeros(padding)))
#     return r, mask

def _one_hot(x: np.ndarray, num_class: int) -> np.ndarray:
    x = x - x.min(None)
    return np.squeeze(np.eye(num_class)[([x],)])


def _get_year(df: pd.DataFrame) -> np.ndarray:
    return df['year'].to_numpy()


def get_year_onehot(df: pd.DataFrame) -> np.ndarray:
    return _one_hot(_get_year(df), 2)


def _get_month(df: pd.DataFrame) -> np.ndarray:
    return df['month'].to_numpy()


def get_month_onehot(df: pd.DataFrame) -> np.ndarray:
    return _one_hot(_get_month(df), 12)


def _get_day(df: pd.DataFrame) -> np.ndarray:
    return df['day'].to_numpy()


def get_day_onehot(df: pd.DataFrame) -> np.ndarray:
    return _one_hot(_get_day(df), 31)


def _get_weekday(df: pd.DataFrame) -> np.ndarray:
    return df['weekday'].to_numpy()


def get_weekday_onehot(df: pd.DataFrame) -> np.ndarray:
    return _one_hot(_get_weekday(df), 7)


def _get_hour(df: pd.DataFrame) -> np.ndarray:
    return df['hour'].to_numpy()


def get_hour_onehot(df: pd.DataFrame) -> np.ndarray:
    return _one_hot(_get_hour(df), 24)


def get_speed(df: pd.DataFrame, shift=0) -> np.ndarray:
    temp = df['speed'].fillna(0).to_numpy()
    if shift == 0:
        return temp.reshape((-1,1))
    elif shift > 0:
        return np.concatenate((np.zeros(shift), temp[:-shift])).reshape((-1,1))
    elif shift < 0:
        return np.concatenate((temp[-shift:], np.zeros(-shift))).reshape((-1,1))
    else:
        return temp.reshape((-1,1))


def get_other_features(fname):
    if fname not in ['wind_dir']:
        return lambda df: df[fname].to_numpy().reshape((-1, 1))
    elif fname == 'wind_dir':
        def f(df):
            dir = df[fname].to_numpy()
            sins = np.sin(dir/180 * np.pi)
            coss = np.cos(dir/180 * np.pi)
            return np.vstack((sins, coss)).T
        return f


def get_feature_size(features: list) -> list:
    def f_str_to_size(s):
        if type(s) == int or re.match('-?\d+', s):
            return 1
        elif s == "year":
            return 2
        elif s == "month":
            return 12
        elif s == "day":
            return 31
        elif s == "weekday":
            return 7
        elif s == "hour":
            return 24
        elif s == "wind_dir":
            return 2
        else:
            return 1

    return [f_str_to_size(s) for s in features]

def get_speed_functional(shift=0):
    """
    +ve: previous, -ve: nex
    :param shift:
    :return:
    """
    def t(df: pd.DataFrame) -> np.ndarray:
        temp = df['speed'].to_numpy()
        if shift == 0:
            return temp.reshape((-1, 1))
        elif shift > 0:
            return np.concatenate((np.full(shift, np.nan), temp[:-shift])).reshape((-1, 1))
        elif shift < 0:
            return np.concatenate((temp[-shift:], np.full(-shift, np.nan))).reshape((-1, 1))
        else:
            return temp.reshape((-1, 1))

    return t


def composite(*args):
    return np.concatenate(args, axis=1)


def composite_functional(df, *args):
    return np.hstack(list(map(lambda x: x(df), args)))
