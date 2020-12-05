import os
import pandas as pd

dir = os.path.dirname(__file__)


def load_train(fill_end=False, complete=True, interpolate=1, fill_missing_2017=True) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(dir, 'train.csv'), parse_dates=['date'], dayfirst=True)
    df = df.set_index('date')
    if fill_missing_2017:
        df = pd.concat((df['2017'].resample('H').interpolate('polynomial', order=interpolate), df['2018']))
    if fill_end:
        if not pd.to_datetime('2018-12-31 23:00', dayfirst=True, yearfirst=True) in df.index:
            df.loc[pd.to_datetime('2018-12-31 23:00', dayfirst=True, yearfirst=True)] = float('nan')
        if not pd.to_datetime('2018-1-1 00:00', dayfirst=True, yearfirst=True) in df.index:
            df.loc[pd.to_datetime('2018-1-1 00:00', dayfirst=True, yearfirst=True)] = float('nan')
    if complete:
        if interpolate == 0:
            df = df.resample('H').last()
        else:
            df = df.resample('H').interpolate('polynomial', order=interpolate)
    return df


def load_test(fill_end=False, complete=False, interpolate=0) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(dir, 'test.csv'), parse_dates=['date'], dayfirst=True)
    df = df.set_index('date')
    df['speed'] = float('nan')
    if fill_end:
        if not pd.to_datetime('2018-12-31 23:00') in df.index:
            df.loc[pd.to_datetime('2018-12-31 23:00')] = float('nan')
        if not pd.to_datetime('2018-1-1 00:00') in df.index:
            df.loc[pd.to_datetime('2018-1-1 00:00')] = float('nan')
    if complete:
        if interpolate == 0:
            df = df.resample('H').last()
        else:
            df = df.resample('H').interpolate('polynomial', order=interpolate)
    return df

def load_complete() -> pd.DataFrame:
    train = load_train(fill_end=False, complete=False)
    b = pd.DataFrame(pd.date_range(pd.to_datetime('1/1/2017 00:00', dayfirst=True, yearfirst=True), pd.to_datetime('12/31/2018 23:00', dayfirst=True, yearfirst=True), freq='H'), columns=['date'])
    b = b.set_index('date')
    b = b.merge(train, 'left', left_index=True, right_index=True)
    b['id'] = list(range(len(b)))
    return b

def load_weather(data, parse_trace=0) -> pd.DataFrame:
    data_weather = pd.read_csv(os.path.join(dir, 'weather.csv'), parse_dates=['date'], dayfirst=False)
    data_weather = data_weather.set_index('date')
    data_weather.loc[pd.to_datetime('2018-12-31 23:00')] = data_weather.loc[pd.to_datetime('2018-12-31 00:00')]
    data_weather = data_weather.resample('H').ffill()
    data_weather['rainfall'] = data_weather['rainfall'].replace('Trace', parse_trace).astype(float)
    data = data.merge(data_weather, 'left', left_index=True, right_index=True)
    return data


def parse_date_into_cols(df: pd.DataFrame):
    df['date'] = df.index
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['hour'] = df['date'].apply(lambda x: x.hour)
    df['weekday'] = df['date'].apply(lambda x: x.isoweekday())
    df = df.drop('date', axis=1)
    return df


def test_out(df: pd.DataFrame, fname: str):
    df = df[['id', 'speed']]
    df.to_csv(os.path.join(dir, '..', 'out', fname), index=False)
    return None


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    print("Working Directory for data", os.path.abspath('.'))
    data_all = load_weather(parse_date_into_cols(load_complete()), 0)
