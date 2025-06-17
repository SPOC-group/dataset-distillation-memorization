import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np

from dataclasses import dataclass
import dataclasses
import wandb


device = 'cpu'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class DataConfig:
    num_samples_per_class: int = 100
    num_samples_per_class_val: int = 20
    input_dim: int = 1000
    num_labels: int = 100
    
    @property
    def data_seed(self):
        if not hasattr(self, '_data_seed'):
            self._data_seed = np.random.randint(0, 1000)
        return self._data_seed
    
    
@dataclass
class TeacherConfig:
    learning_rate: float = 0.001
    hidden_dim: int = 500
    num_epochs: int = 100

    model_name: str = 'two_layer_network'
    activation: str = 'relu'
    weight_decay: float = 0.0
    bias: bool = True
    early_stopping: bool = False
    
    @property
    def train_seed(self):
        if not hasattr(self, '_train_seed'):
            self._train_seed = np.random.randint(0, 1000)
        return self._train_seed
    
    @property
    def model_seed(self):
        if not hasattr(self, '_model_seed'):
            self._model_seed = np.random.randint(0, 1000)
        return self._model_seed
    
    
@dataclass
class StudentConfig:
    learning_rate: float = 0.0001
    hidden_dim: int = 1000
    num_epochs: int = 4000
    
    train_frac: float = 0.8
    temperature: float = 20.0
    bias: bool = True
    
    soft_label_treatment: str = None
    soft_label_filtering: str = None
    shuffle_input_intra_class: bool = False

    model_name: str = 'two_layer_network'
    activation: str = 'relu'
    weight_decay: float = 0.0
    
    @property
    def train_seed(self):
        if not hasattr(self, '_train_seed'):
            self._train_seed = np.random.randint(0, 1000)
        return self._train_seed
    
    
def create_dataset(config):
    # doubling the number of labels can be  compensated by double of input labels
    num_samples = config.num_samples_per_class * config.num_labels

    rs_data_train = np.random.RandomState(config.data_seed)
    rs_data_val = np.random.RandomState(config.data_seed + 1)

    X_train = rs_data_train.randn(num_samples, config.input_dim) / np.sqrt(config.input_dim) # Random points in [-1,1]
    # use gaussian distribution instead of uniform distribution
    #X_random = np.random.randn(num_samples, input_dim)  # Random points in [-1,1]
    # list with num_labels random labels that all appear the same number of times
    y_train = np.concatenate([np.full(config.num_samples_per_class, label) for label in range(config.num_labels)])
    y_train = rs_data_train.permutation(y_train)


    X_val = rs_data_val.rand(config.num_samples_per_class_val * config.num_labels, config.input_dim)
    y_val = np.concatenate([np.full(config.num_samples_per_class_val, label) for label in range(config.num_labels)])
    y_val = rs_data_val.permutation(y_val)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
        
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    return X_train, y_train, X_val, y_val


class TwoLayerNet(nn.Module):
    def __init__(self, teacher_config, data_config):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(data_config.input_dim, teacher_config.hidden_dim,bias=teacher_config.bias)
        self.fc2 = nn.Linear(teacher_config.hidden_dim, teacher_config.hidden_dim,bias=teacher_config.bias)
        self.fc3 = nn.Linear(teacher_config.hidden_dim, data_config.num_labels,bias=teacher_config.bias)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)
        return x
    
class OneLayerNet(nn.Module):
    def __init__(self, teacher_config, data_config):
        super(OneLayerNet, self).__init__()
        self.fc1 = nn.Linear(data_config.input_dim, teacher_config.hidden_dim,bias=teacher_config.bias)
        self.fc2 = nn.Linear(teacher_config.hidden_dim, data_config.num_labels,bias=teacher_config.bias)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.fc2(x)
        return x
    
class RegressionNet(nn.Module):
    def __init__(self, teacher_config, data_config):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(data_config.input_dim, data_config.num_labels,bias=teacher_config.bias)

    def forward(self, x):
        return self.fc1(x)
    
class LogisticRegressionNet(nn.Module):
    def __init__(self, teacher_config, data_config):
        super(LogisticRegressionNet, self).__init__()
        self.fc1 = nn.Linear(data_config.input_dim,1,bias=teacher_config.bias)

    def forward(self, x):
        return self.fc1(x)
    
def weight_and_bias_norms(model):
    norms = []
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            norms.append(param.norm().item())
    return norms
    
def named_weight_and_bias_norms(model,prefix=None):
    norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            norms[name if prefix is None else prefix + name] = param.norm().item()
    return norms
    
models = {
    'two_layer_network': TwoLayerNet,
    'one_layer_network': OneLayerNet,
    'regression': RegressionNet,
    'logistic_regression': LogisticRegressionNet
}



def train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val):
    torch.manual_seed(teacher_config.model_seed)
    model = models[teacher_config.model_name](teacher_config,  data_config)
    

    criterion = nn.BCEWithLogitsLoss()

    torch.manual_seed(teacher_config.train_seed)
    if teacher_config.weight_decay > 0:
        optimizer = optim.Adam(model.parameters(), lr=teacher_config.learning_rate, weight_decay=teacher_config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=teacher_config.learning_rate) # 0.001
    
    model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(teacher_config.num_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(1), y_train.float())
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        # Compute accuracy
        with torch.no_grad():
            # round to 0 or 1 via sigmoid then clip to nearest int
            train_pred = torch.sigmoid(output)
            train_pred = (train_pred > 0.5).long().squeeze(1)
            train_acc = (train_pred == y_train).float().mean().item()
            val_output = model(X_val)
            val_loss = criterion(val_output.squeeze(1), y_val.float())
            val_pred = torch.sigmoid(val_output)
            val_pred = (val_pred > 0.5).long().squeeze(1)
            val_acc = (val_pred == y_val).float().mean().item()
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}')
            
            if teacher_config.early_stopping:
                if train_acc == 1.0:
                    print('Early stopping at epoch:', epoch)
                    break

            #wandb.log({'teacher_train_loss': loss.item(), 'teacher_train_acc': train_acc, 'teacher_val_loss': val_loss.item(), 'teacher_val_acc': val_acc})
            

    return model, {
        'teacher_train_losses': train_losses,
        'teacher_train_accuracies': train_accuracies,
        'teacher_val_accuracies': val_accuracies,
        'teacher_train_loss': train_losses[-1],
        'teacher_train_acc': train_accuracies[-1],
        'teacher_val_acc': val_accuracies[-1],
        'teacher_weight_norms': named_weight_and_bias_norms(model)
    }
    

class LinearModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)
    
def run_student_exp_linear_model(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val):
    
    soft_labels = teacher_model(X_train)
    print(soft_labels.shape)
    train_X, train_y, test_X, test_y, soft_labels_train, soft_labels_test = [], [], [], [], [], []

    for label in range(data_config.num_labels):
        indices = torch.where(y_train == label)[0]  # Get indices for the label
        train_size = int(student_config.train_frac * data_config.num_samples_per_class)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        if student_config.shuffle_input_intra_class:
            label_perm = torch.randperm(len(train_indices))
            train_X.append(X_train[train_indices[label_perm]])
        else:
            train_X.append(X_train[train_indices])
        
        train_y.append(y_train[train_indices])
        
        test_X.append(X_train[test_indices])
        test_y.append(y_train[test_indices])
        
        soft_labels_train.append(soft_labels[train_indices])
        soft_labels_test.append(soft_labels[test_indices])

    # Concatenate into final datasets
    X_transfer_train = torch.cat(train_X)
    y_transfer_train = torch.cat(train_y)
    X_transfer_test = torch.cat(test_X)
    y_transfer_test = torch.cat(test_y)
    soft_labels_train = torch.cat(soft_labels_train)
    soft_labels_test = torch.cat(soft_labels_test)
    
    # train a linear model via least squares
    model = LogisticRegressionNet(student_config,  data_config)
    model.to(device)
    X_transfer_train = X_transfer_train.to(device)
    soft_labels_train = soft_labels_train.to(device)
    X_transfer_test = X_transfer_test.to(device)
    soft_labels_test = soft_labels_test.to(device)
    y_transfer_test = y_transfer_test.to(device)
    y_transfer_train = y_transfer_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    # Compute the pseudo-inverse of the training data
    pseudo_inverse = torch.linalg.pinv(X_transfer_train)
    # Compute the weights using the pseudo-inverse and soft labels

    #print(pseudo_inverse @ soft_labels_train)
    with torch.no_grad():
        model.fc1.weight.copy_((pseudo_inverse @ soft_labels_train).T)
    # no bias
    #model.linear.bias.copy_(soft_labels_train.mean(dim=0))
    # Compute the accuracy on the training and test sets
    
    with torch.no_grad():
        train_mse = nn.MSELoss()(model(X_transfer_train), soft_labels_train)
        test_mse = nn.MSELoss()(model(X_transfer_test), soft_labels_test)
        
        train_pred = torch.sigmoid(model(X_transfer_train))
        train_pred = (train_pred > 0.5).long().squeeze(1)
        train_acc = (train_pred == y_transfer_train).float().mean().item()

        
        teacher_train_pred = torch.sigmoid(soft_labels_train)
        teacher_train_pred = (teacher_train_pred > 0.5).long().squeeze(1)
        match_teacher_acc_train = (teacher_train_pred == train_pred).float().mean().item()

        test_pred = torch.sigmoid(model(X_transfer_test))
        test_pred = (test_pred > 0.5).long().squeeze(1)
        test_acc = (test_pred == y_transfer_test).float().mean().item()
        
        teacher_test_pred = torch.sigmoid(soft_labels_test)
        teacher_test_pred = (teacher_test_pred > 0.5).long().squeeze(1)
        match_teacher_acc_test = (teacher_test_pred == test_pred).float().mean().item()
        
        # also compare to original test loss
        test_pred_orig = torch.sigmoid(model(X_val))
        test_pred_orig = (test_pred_orig > 0.5).long().squeeze(1)
        test_acc_orig = (test_pred_orig == y_val).float().mean().item()
        
        # check the test loss specifically on 0 labels
        zeros_mask = y_transfer_test == 0
        test_acc_zeros = (test_pred[zeros_mask] == 0).float().mean().item()
    # Store metrics
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    test_accuracies_orig = []
    test_accuracies_zeros = []
    match_teacher_accs_train = []
    match_teacher_accs_test = []
    train_mses = []
    test_mses = []
    train_mses.append(train_mse.item())
    test_mses.append(test_mse.item())
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    test_accuracies_orig.append(test_acc_orig)
    test_accuracies_zeros.append(test_acc_zeros)
    match_teacher_accs_train.append(match_teacher_acc_train)
    match_teacher_accs_test.append(match_teacher_acc_test)

    
    return model, {
        #'train_losses': (train_losses),
        'train_accuracies': (train_accuracies),
        #'test_losses': (test_losses),
        'test_mses': (test_mses),
        'train_mses': (train_mses),
        'test_accuracies': (test_accuracies),
        'test_accuracies_orig': (test_accuracies_orig),
        'test_accuracies_zeros': (test_accuracies_zeros),
        'match_teacher_accs_train': (match_teacher_accs_train),
        'match_teacher_accs_test': (match_teacher_accs_test),}
    
    
    
def combine_dataclasses(data_config, teacher_config, student_config, extra_data):
    
    new_data_config = { 'data_' + key : item  for key, item in dataclasses.asdict(data_config).items()}
    new_teacher_config = { 'teacher_' + key : item  for key, item in dataclasses.asdict(teacher_config).items()}
    new_student_config = { 'student_' + key : item  for key, item in dataclasses.asdict(student_config).items()}
    
    return {**new_data_config, **new_teacher_config, **new_student_config, **extra_data}
