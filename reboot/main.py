from data.load import *
import torch
from date_feature_extractor import *
import re
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import Model_builder
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, train_test_split
from datetime import datetime
import logging

"""
features:
'features': ['month', 'day', 'weekday', 'hour', 1, 'mean_pressure', 'temp_max', 'temp_mean', 'temp_min', 'dew_point', 'humidity', 'cloud', 'rainfall', 'sunshine', 'wind_dir', 'wind_speed'],
"""

params = {
    'model': 'dnn_relu',
    'layer': [768, 1],
    'features': ['month', 'day', 'weekday', 'hour', 1],
    # 'features': ['month', 'day', 'weekday', 'hour', 1, 'mean_pressure', 'temp_max', 'temp_mean', 'temp_min', 'dew_point', 'humidity', 'cloud', 'rainfall', 'sunshine', 'wind_dir', 'wind_speed'],
    'speed_normalizer': 100,
    'config': {'dropout': [0.25], 'regul': 0.05},
    'learning_rate': 1e-3,
    'optim': 'adamw',
    'epochs': 50,
    'scheduler': 'milestone',
    'milestone': [],
    'eval': 2,
    'batch_size': 8,
    'k-fold': None,
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


def model_factory(model_name, feature_size, layer, config) -> torch.nn.Module:
    model_class = get_model(model_name)
    model = model_class(feature_size, layer, **config)
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


def save(model: torch.nn.Module, PROJ_ID: str = ""):
    path = f'{PROJ_ID}.pt'
    path = os.path.join('model', path)
    torch.save(model.state_dict(), path)
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

    """As we are training and eval with no windowing, these functions are equal"""

    def eval_no_window(net: torch.nn.Module, test_set: DataLoader, verbose=True):
        net.eval()
        total_loss = 0
        nun_item = 0
        y_running = []
        with tqdm(total=len(test_set), disable=not verbose) as pbar:
            for i, (x, y) in enumerate(test_set):
                y_ = net(x)
                y_l = y_.cpu().detach().numpy().reshape(-1).tolist()
                y_running += y_l
                total_loss += loss_f(y_ * params['speed_normalizer'], y * params['speed_normalizer']).item()
                nun_item += len(y_l)
                pbar.update()
        if verbose:
            print(f"Eval Epoch: loss={total_loss:5.3f}, avg={total_loss / nun_item:5.3f}")
        return total_loss, nun_item, y_running

    def eval_w_window(net: torch.nn.Module, test_set: DataLoader, verbose=True):
        net.eval()
        total_loss = 0
        nun_item = 0
        y_running = []
        with tqdm(total=len(test_set), disable=not verbose) as pbar:
            for i, (x, y) in enumerate(test_set):
                y_ = net(x)
                y_l = y_.cpu().detach().numpy().reshape(-1).tolist()
                y_running += y_l
                total_loss += loss_f(y_ * params['speed_normalizer'], y * params['speed_normalizer']).item()
                nun_item += len(y_l)
                pbar.update()
        if verbose:
            print(f"Eval Epoch: loss={total_loss:5.3f}, avg={total_loss / nun_item:5.3f}")
        return total_loss, nun_item, y_running

    return eval_w_window if flag_window else eval_no_window


def infer_func_builder(features: list, data_all=None):
    feature_is_prev_speed = np.array(
        [(type(s) == int) or (type(s) == str and re.match('-?\d+', s) is not None) for s in features])
    flag_window = feature_is_prev_speed.any()
    feature_ele_size = np.array(get_feature_size(features))
    feature_is_speed_index = np.argwhere(feature_is_prev_speed)[:, 0]
    feature_index = np.array(
        [feature_ele_size[:i].sum() for i in feature_is_speed_index])  # a list of index where the feature is at
    feature_shift = np.array(features)[feature_is_speed_index].astype(int)  # a list of amount of shift

    def infer_no_window(net: torch.nn.Module, data_target: pd.DataFrame, verbose=True) -> np.ndarray:
        net.eval()
        target_features = feature_factory(data_target, params['features'])
        if USE_GPU:
            target_features = torch.tensor(target_features, dtype=torch.float32).cuda()
        else:
            target_features = torch.tensor(target_features, dtype=torch.float32)
        with tqdm(total=1, disable=not verbose) as pbar:
            target_y = net(target_features)
            target_speed = target_y.cpu().detach().numpy().reshape(-1)
            pbar.update()
        return target_speed

    def infer_w_window(net: torch.nn.Module, data_target: pd.DataFrame, verbose=True) -> np.ndarray:
        net.eval()
        all_features = feature_factory(data_all, params['features'])
        all_speed = data_all['speed'].to_numpy()
        if USE_GPU:
            all_features = torch.tensor(all_features, dtype=torch.float32).cuda()
            all_speed = torch.tensor(all_speed, dtype=torch.float32).cuda()
        else:
            all_features = torch.tensor(all_features, dtype=torch.float32)
            all_speed = torch.tensor(all_speed, dtype=torch.float32)
        with tqdm(total=len(data_all), disable=not verbose) as pbar:
            for i in range(len(data_all)):
                x = all_features[i]
                y = all_speed[i]
                if not torch.isnan(y).item():
                    pbar.update()
                    continue
                if torch.isnan(x).any().item():
                    raise ValueError("Cannot use nan in x to infer y")
                y_ = net(x.reshape((1, -1)))
                all_speed[i] = y_
                for j, s in zip(feature_index, feature_shift):
                    if 0 <= i + s < len(data_all):
                        all_features[i + s][j] = y_
                pbar.update()
        r_data = data_all.copy()
        r_data['speed'] = all_speed.cpu().detach().numpy().reshape(-1)
        r = data_target[['id']].merge(r_data[['speed']], left_index=True, right_index=True)
        return r['speed'].to_numpy()

    return infer_w_window if flag_window else infer_no_window


def main(params, USE_GPU, INFER, verbose=False):
    date_now = datetime.now().strftime('%m%d%H%M')
    PROJ_ID = "-".join(
        [params['model'], '_'.join([str(i) for i in params['layer']]),
         '_'.join([i[0] + i[-1] if type(i) == str else str(i) for i in params['features']]),
         params['optim'], str(params['speed_normalizer']), str(params['epochs']),
         '_'.join([str(i) for i in params['milestone']]), '_'.join(
            [f"{k}_{v}" if type(v) != list else f"{k}_{'_'.join([str(i) for i in v])}" for k, v in
             params['config'].items()]), f"lr_{params['learning_rate']}"])

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
    training_dataloader = None
    testing_dataloader = None
    dataset = None
    if USE_GPU:
        dataset = TensorDataset(torch.Tensor(training_features).cuda(),
                                torch.Tensor(training_truth).cuda().reshape((-1, 1)))
    else:
        dataset = TensorDataset(torch.Tensor(training_features), torch.Tensor(training_truth))
    training_len = int(len(dataset) * 0.8)
    testing_len = len(dataset) - training_len
    training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [training_len, testing_len])
    training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=params['batch_size'], num_workers=0)
    testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=params['batch_size'], num_workers=0)

    # model
    print('--Constructing Model--')
    net = model_factory(params['model'], feature_size, params['layer'], params['config'])
    if USE_GPU:
        net = net.cuda()

    # loss
    print('--Constructing Loss--')
    loss_f = loss_factory('mse')

    # optim
    print('--Constructing Optimizer--')
    optim = optim_factory(params['optim'], net.parameters(), params['learning_rate'])

    # scheduler
    print('--Constructing Scheduler--')
    if params['scheduler']:
        scheduler = scheduler_factory(optim, params['milestone'])

    # eval function
    print('--Contructing Eval Func')
    eval_f = eval_func_builder(params['features'], loss_f, data_all)

    # writer = SummaryWriter(f"runs/{PROJ_ID}_{date_now}")
    # writer.add_graph(net, next(iter(testing_dataloader))[0], True)
    #
    # print('--Training--')
    # for epoch in range(params['epochs']):
    #     # training
    #     net.train()
    #     running_loss = 0
    #     with tqdm(total=len(training_dataloader), disable=not verbose) as pbar:
    #         for i, (x, y) in enumerate(training_dataloader):
    #             optim.zero_grad()
    #             y_ = net(x)
    #             regul = 0
    #             for p in net.parameters():
    #                 regul += p.norm(2)
    #             loss = loss_f(y_ * params['speed_normalizer'], y * params['speed_normalizer'])
    #             running_loss += loss
    #             loss += params['config']['regul'] * regul
    #             loss.backward()
    #             optim.step()
    #             pbar.update()
    #     if verbose:
    #         print(f"Epoch {epoch:02d}: loss={running_loss:5.3f}, avg={running_loss / len(training_dataloader):5.3f}")
    #     writer.add_scalar('Loss/train', running_loss / len(training_dataloader), epoch)
    #     if params['scheduler']:
    #         scheduler.step()
    #     # eval
    #     if epoch % params['eval'] == params['eval'] - 1:
    #         eval_total_loss, num_eval_item, _ = eval_f(net, testing_dataloader, verbose=verbose)
    #         writer.add_scalar('Loss/eval', eval_total_loss / num_eval_item, epoch // params['eval'] + 1)
    # save(net, f"{PROJ_ID}_{date_now}")
    #
    # eval_total_loss, num_eval_item, _ = eval_f(net, testing_dataloader, verbose=verbose)
    if INFER:
        print("INFER_retraining")
        training_dataloader = DataLoader(dataset, shuffle=True, batch_size=params['batch_size'], num_workers=0)

        # model
        print('--Constructing Model--')
        net = model_factory(params['model'], feature_size, params['layer'], params['config'])
        if USE_GPU:
            net = net.cuda()
        net_name = "model/final_model.pt"
        net.load_state_dict(torch.load(net_name))
        infer_f = infer_func_builder(params['features'], data_all)
        data_target['speed'] = infer_f(net, data_target)*params['speed_normalizer']
        data_target[['id', 'speed']].to_csv(f"out/{PROJ_ID}_{date_now}.csv", index=False)
        print("csv:", f"out/{PROJ_ID}_{date_now}.csv")

    print(PROJ_ID)

    print("Finished:", PROJ_ID)
    print("*" * 20)
    return PROJ_ID


if __name__ == '__main__':
    main(params, USE_GPU, INFER)
