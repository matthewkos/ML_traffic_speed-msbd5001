from data.load import *
import torch
from date_feature_extractor import *
import re
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sk_model import Model_builder, SKModel
from joblib import dump
from datetime import datetime
params = {
    'model': 'skxgbdart',
    # 'features': ['month', 'day', 'weekday', 'hour', 1],
    'features': ['month', 'day', 'weekday', 'hour', 1, 'mean_pressure', 'temp_max', 'temp_mean', 'temp_min',
                 'dew_point', 'humidity', 'cloud', 'rainfall', 'sunshine', 'wind_dir', 'wind_speed'],
    'speed_normalizer': 1,
    'config': {'n': 768},
}

USE_GPU = torch.cuda.is_available()
INFER = True


def feature_factory(df: pd.DataFrame, features: list) -> pd.DataFrame:
    def f_str_to_func(s):
        if type(s) == int or re.match('-?\d+', s):
            s = int(s)
            return get_speed_functional(s)
        elif s == "year":
            return get_year_onehot
        elif s == "month":
            return get_month_onehot
        elif s == "day":
            return get_day_onehot
        elif s == "weekday":
            return get_weekday_onehot
        elif s == "hour":
            return get_hour_onehot
        else:
            return get_other_features(s)

    return composite_functional(df, *[f_str_to_func(s) for s in features])


def get_model(model_name):
    return getattr(Model_builder, model_name)()


def model_factory(model_name, config) -> SKModel:
    model_class = get_model(model_name)
    model = model_class(**config)
    return model


def optim_factory(optim_name, params, lr):
    def f_str_to_func(s):
        if optim_name == 'adam':
            return torch.optim.Adam
        elif optim_name == 'adamw':
            return torch.optim.AdamW
        elif optim_name == 'adamax':
            return torch.optim.Adamax
        elif optim_name == 'rmsprop':
            return torch.optim.RMSprop
        elif optim_name == 'sgd':
            return torch.optim.SGD
        else:
            raise ValueError(f"{s} is not a valid optim")

    return f_str_to_func(optim_name)(params, lr)


def scheduler_factory(opti, milestones):
    return torch.optim.lr_scheduler.MultiStepLR(opti, milestones)


def loss_factory(loss_name):
    if loss_name == 'mse':
        return torch.nn.MSELoss()


def save(model: SKModel, PROJ_ID: str = ""):
    path = f'{PROJ_ID}.joblib'
    path = os.path.join('model', path)
    dump(model, path)
    print("Model saved:", path)


def load(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)


def eval_func_builder(features: list, loss_f, data_all=None):
    feature_is_prev_speed = np.array(
        [(type(s) == int) or (type(s) == str and re.match('-?\d+', s) is not None) for s in features])
    flag_window = feature_is_prev_speed.any()
    feature_ele_size = np.array(get_feature_size(features))
    feature_is_speed_index = np.argwhere(feature_is_prev_speed)[:, 0]
    feature_index = np.array(
        [feature_ele_size[:i].sum() for i in feature_is_speed_index])  # a list of index where the feature is at
    feature_shift = np.array(features)[feature_is_speed_index].astype(int)  # a list of amount of shift

    def eval_w_window(net: SKModel, testing_x: np.ndarray, testing_y: np.ndarray,verbose=True):
        y_ = net(testing_x)
        loss = loss_f(torch.tensor(y_) * params['speed_normalizer'],
                      torch.tensor(testing_y) * params['speed_normalizer'])
        print(f"Eval Epoch: loss={loss:5.3f}, avg={loss / len(testing_y):5.3f}")
        return loss, len(testing_y)

    def eval_no_window(net: SKModel, testing_x: np.ndarray, testing_y: np.ndarray,verbose=True):
        y_ = net(testing_x)
        loss = loss_f(torch.tensor(y_) * params['speed_normalizer'],
                      torch.tensor(testing_y) * params['speed_normalizer'])
        print(f"Eval Epoch: loss={loss:5.3f}, avg={loss / len(testing_y):5.3f}")
        return loss, len(testing_y)

    return eval_w_window if flag_window else eval_no_window()


def infer_func_builder(features: list, data_all=None):
    feature_is_prev_speed = np.array(
        [(type(s) == int) or (type(s) == str and re.match('-?\d+', s) is not None) for s in features])
    flag_window = feature_is_prev_speed.any()
    feature_ele_size = np.array(get_feature_size(features))
    feature_is_speed_index = np.argwhere(feature_is_prev_speed)[:, 0]
    feature_index = np.array(
        [feature_ele_size[:i].sum() for i in feature_is_speed_index])  # a list of index where the feature is at
    feature_shift = np.array(features)[feature_is_speed_index].astype(int)  # a list of amount of shift

    def infer_no_window(net: SKModel, data_target: pd.DataFrame) -> np.ndarray:
        target_features = feature_factory(data_target, params['features'])
        with tqdm(total=1) as pbar:
            target_y = net(target_features)
            target_speed = target_y.reshape(-1)
            pbar.update()
        return target_speed

    def infer_w_window(net: torch.nn.Module, data_target: pd.DataFrame) -> np.ndarray:
        all_features = feature_factory(data_all, params['features'])
        all_speed = data_all['speed'].to_numpy()
        with tqdm(total=len(data_all)) as pbar:
            for i in range(len(data_all)):
                x = all_features[i]
                y = all_speed[i]
                if not np.isnan(y):
                    pbar.update()
                    continue
                if np.isnan(x).any():
                    raise ValueError("Cannot use nan in x to infer y")
                y_ = net(x.reshape(1, -1))
                all_speed[i] = y_
                for j, s in zip(feature_index, feature_shift):
                    if 0 <= i + s < len(data_all):
                        all_features[i + s][j] = y_
                pbar.update()
        r_data = data_all.copy()
        r_data['speed'] = all_speed.reshape(-1)
        r = data_target[['id']].merge(r_data[['speed']], left_index=True, right_index=True)
        return r['speed'].to_numpy()

    return infer_w_window if flag_window else infer_no_window


def main(params, USE_GPU, INFER, verbose=False):
    date_now = datetime.now().strftime('%m%d%H%M')

    PROJ_ID = "-".join(
        ['sk', params['model'], '_'.join([i[0] + i[-1] if type(i) == str else str(i) for i in params['features']]),
         str(params['speed_normalizer']),
         '_'.join([f"{k}_{v}" if type(v) != list else f"{k}_{'_'.join([str(i) for i in v])}" for k, v in
                   params['config'].items()])])

    print("Project:", PROJ_ID)

    print("Starting Project:", PROJ_ID)
    print('--Preparing data--')
    # data_train = parse_date_into_cols(load_train(False, False))
    data_target = load_weather(parse_date_into_cols(load_test()))
    data_all = load_weather(parse_date_into_cols(load_complete()))

    # Data
    data_all['speed'] = data_all['speed'] / params['speed_normalizer']
    training_features = feature_factory(data_all, params['features'])
    valid_training_index = ~np.isnan(training_features).any(axis=1) & ~np.isnan(data_all['speed'].to_numpy())
    training_truth = data_all['speed'][valid_training_index]
    training_features = training_features[valid_training_index]
    feature_size = training_features.shape[-1]
    target_features = feature_factory(data_target, params['features'])
    dataset = TensorDataset(torch.Tensor(training_features), torch.Tensor(training_truth))
    training_len = int(len(dataset) * 0.8)
    testing_len = len(dataset) - training_len
    training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [training_len, testing_len])
    training_x = np.vstack([x.numpy() for x, y in training_dataset])
    training_y = np.array([y.numpy() for x, y in training_dataset])
    testing_x = np.vstack([x.numpy() for x, y in testing_dataset])
    testing_y = np.array([y.numpy() for x, y in testing_dataset])

    # model
    print('--Constructing Model--')
    net = model_factory(params['model'], params['config'])

    # loss
    print('--Constructing Loss--')
    loss_f = loss_factory('mse')

    # eval function
    print('--Contructing Eval Func')
    eval_f = eval_func_builder(params['features'], loss_f, data_all)

    writer = SummaryWriter(f"runs/{PROJ_ID}_{date_now}")

    print('--Training--')
    y_ = net.train(training_x, training_y)
    loss = loss_f(torch.tensor(y_) * params['speed_normalizer'], torch.tensor(training_y) * params['speed_normalizer'])
    print(f"Training: loss={loss:5.3f}, avg={loss / len(training_y):5.3f}")
    writer.add_scalar('Loss/train', loss / len(training_y), 0)
    print('--Eval--')
    eval_total_loss, num_eval_item = eval_f(net, testing_x, testing_y)
    writer.add_scalar('Loss/eval', eval_total_loss / num_eval_item, 0)

    save(net, f"{PROJ_ID}_{date_now}")

    if INFER:
        print("INFER_retraining")
        training_x = np.vstack([x.numpy() for x, y in training_dataset])
        training_y = np.array([y.numpy() for x, y in training_dataset])
        net = model_factory(params['model'], params['config'])
        y_ = net.train(training_x, training_y)
        loss = loss_f(torch.tensor(y_) * params['speed_normalizer'],
                      torch.tensor(training_y) * params['speed_normalizer'])
        print(f"Training: loss={loss:5.3f}, avg={loss / len(training_y):5.3f}")
        eval_total_loss, num_eval_item = eval_f(net, training_x, training_y)

        infer_f = infer_func_builder(params['features'], data_all)
        data_target['speed'] = infer_f(net, data_target)
        data_target[['id', 'speed']].to_csv(f"out/{PROJ_ID}_{date_now}.csv", index=False)

    print("Finished:", PROJ_ID)
    print("*" * 20)
    return PROJ_ID, eval_total_loss / num_eval_item


if __name__ == '__main__':
    main(params, USE_GPU, INFER)
