from trains import Task
from data.load import *
import torch
from date_feature_extractor import *
import re
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model import Model_builder
from torch.utils.tensorboard import SummaryWriter

"""
features:
month, day, weekday, hour, 1, 2, 3
"""
params = {
    'model': 'dnn_relu',
    'layer': [256, 1],
    'features': ['month', 'day', 'weekday', 'hour'],
    # 'features': ['month', 'day', 'weekday', 'hour', 'mean_pressure', 'temp_max', 'temp_mean', 'temp_min', 'dew_point',
    #    'humidity', 'cloud', 'rainfall', 'sunshine', 'wind_dir', 'wind_speed'],
    'speed_normalizer': 1,
    'config': {'dropout': [0.25]},
    'learning_rate': 1e-3,
    'optim': 'adam',
    'epochs': 300,
    'scheduler': None,
    'milestone': [],
    'eval': 20,
    'batch_size': 8,
}

PROJ_ID = "-".join(
    [params['model'], '_'.join([str(i) for i in params['layer']]), '_'.join([str(i)[0]+str(i)[-1] for i in params['features']]),
     params['optim'], str(params['speed_normalizer']), str(params['epochs']),
     '_'.join([str(i) for i in params['milestone']]), '_'.join(
        [f"{k}_{v}" if type(v) != list else f"{k}_{'_'.join([str(i) for i in v])}" for k, v in
         params['config'].items()])])

USE_GPU = torch.cuda.is_available()
INFER = True

task = Task.init(project_name="ML_traffic_speed_msbd5001", task_name=PROJ_ID)


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


def save(model: torch.nn.Module, path: str = None):
    if path is None:
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
    flag_infer = feature_is_prev_speed.any()
    feature_ele_size = np.array(get_feature_size(features))
    feature_is_speed_index = np.argwhere(feature_is_prev_speed)[:, 0]
    feature_index = np.array(
        [feature_ele_size[:i].sum() for i in feature_is_speed_index])  # a list of index where the feature is at
    feature_shift = np.array(features)[feature_is_speed_index].astype(int)  # a list of amount of shift

    def eval_no_infer(net: torch.nn.Module, test_set: DataLoader):
        net.eval()
        total_loss = 0
        nun_item = 0
        y_running = []
        with tqdm(total=len(test_set)) as pbar:
            for i, (x, y) in enumerate(test_set):
                y_ = net(x)
                y_l = y_.cpu().detach().numpy().reshape(-1).tolist()
                y_running += y_l
                total_loss += loss_f(y_, y).item()
                nun_item += len(y_l)
                pbar.update()
        print(f"Eval Epoch: loss={total_loss:5.3f}, avg={total_loss / nun_item:5.3f}")
        return total_loss, nun_item, y_running

    def eval_infer(net: torch.nn.Module, test_set: DataLoader):
        net.eval()
        total_loss = 0
        nun_item = 0
        data = data_all.copy()
        y_running = []
        y_infer = []
        with tqdm(total=len(test_set)) as pbar:
            for i in range(len(data_all)):
                if pd.isna(data.iloc[i]['speed']):
                    x = feature_factory(data.iloc[i:i+1], params['features'])
                    x = torch.tensor(x, dtype=torch.float32, device='cuda:0')
                    y_ = net(x)
                    y_l = y_.cpu().detach().numpy().reshape(-1).tolist()[0]
                    data.iloc[i]['speed'] = y_l
                    y_infer.append(y_l)
                else:
                    y = data.iloc[i]['speed']
                    x = feature_factory(data.iloc[i:i + 1], params['features'])
                    x = torch.tensor(x, dtype=torch.float32, device='cuda:0')
                    y_ = net(x)
                    y_l = y_.cpu().detach().numpy().reshape(-1).tolist()[0]
                    y_running.append(y_l)
                    nun_item += 1
                    total_loss += loss_f(y_, y).item()
                pbar.update()
        print(f"Eval Epoch: loss={total_loss:5.3f}, avg={total_loss / nun_item:5.3f}")
        return total_loss, nun_item, (y_infer, data, y_running)
        # # y_running = [.0] * 24
        # y_running = training_set['speed'].to_list()[-24:]
        # with tqdm(total=len(test_set)) as pbar:
        #     for i, (x, y) in enumerate(test_set):
        #         need_replace = (x[:, feature_index] == 0).any().item()
        #         if need_replace:
        #             x_new = x.cpu()
        #             where_shift = feature_shift[(x_new[0, feature_index] == 0).tolist()]
        #             where_index = feature_index[(x_new[0, feature_index] == 0).tolist()]
        #             x_new[:, where_index] = torch.tensor([y_running[-i] for i in where_shift])
        #             x_new = x_new.cuda()
        #         else:
        #             x_new = x
        #         y_ = net(x_new)
        #         y_running.append(y_.item())
        #         if y > 0:
        #             total_loss += loss_f(y_, y).item()
        #             nun_item += 1
        #         pbar.update()
        # print(f"Eval Epoch: loss={total_loss:5.3f}, avg={total_loss / nun_item:5.3f}")
        # return total_loss, nun_item, y_running[24:]

    return eval_infer if flag_infer else eval_no_infer


if __name__ == '__main__':
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
    eval_f = eval_func_builder(params['features'], loss_f)

    writer = SummaryWriter(f"runs/{PROJ_ID}")
    writer.add_graph(net, next(iter(testing_dataloader))[0], True)

    print('--Training--')
    for epoch in range(params['epochs']):
        # training
        net.train()
        running_loss = 0
        with tqdm(total=len(training_dataloader)) as pbar:
            for i, (x, y) in enumerate(training_dataloader):
                optim.zero_grad()
                y_ = net(x)
                regul = 0
                for p in net.parameters():
                    regul += p.norm(2)
                loss = loss_f(y_, y)
                running_loss += loss
                loss += 0.01 * regul
                loss.backward()
                optim.step()
                pbar.update()
        print(f"Epoch {epoch:02d}: loss={running_loss:5.3f}, avg={running_loss / len(training_dataloader):5.3f}")
        writer.add_scalar('Loss/train', running_loss / len(training_dataloader), epoch)
        if params['scheduler']:
            scheduler.step()
        # eval
        if epoch % params['eval'] == params['eval'] - 1:
            eval_total_loss, num_eval_item, _ = eval_f(net, testing_dataloader)
            writer.add_scalar('Loss/eval', eval_total_loss / num_eval_item, epoch // params['eval'] + 1)
    save(net)

    if INFER:
        target_features = torch.tensor(target_features, dtype=torch.float32).cuda()
        target_y = net(target_features)
        target_speed = target_y.cpu().detach().numpy().reshape(-1)
        data_target['speed'] = target_speed
        data_target[['id','speed']].to_csv(f"out/{PROJ_ID}.csv", index=False)
    eval_total_loss, num_eval_item, running_y = eval_f(net, testing_dataloader)
