
import yaml
with open("./hyperparameters.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        a=10
        print(exc)
from dataset.circuit_data import CircuitDataset
from torch.utils.data import DataLoader


train_set = CircuitDataset(params['dataset_dir'], train = False, train_percent=0, size=params['img_size'])
train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True,
                              pin_memory=True, num_workers=params['num_workers'])

mean = 0.0
std = 0.0
nb_samples = 0.0
for data in train_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print(mean,std)