import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataclasses import dataclass
import dataclasses
import wandb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    num_samples = config.num_samples_per_class * config.num_labels

    rs_data_train = np.random.RandomState(config.data_seed)
    rs_data_val = np.random.RandomState(config.data_seed + 1)

    X_train = rs_data_train.randn(num_samples, config.input_dim)
    y_train = np.concatenate([np.full(config.num_samples_per_class, label) for label in range(config.num_labels)])
    y_train = rs_data_train.permutation(y_train)


    X_val = rs_data_val.rand(config.num_samples_per_class_val * config.num_labels, config.input_dim)
    y_val = np.concatenate([np.full(config.num_samples_per_class_val, label) for label in range(config.num_labels)])
    y_val = rs_data_val.permutation(y_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

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
        if teacher_config.activation == 'relu':
            self.activ = nn.ReLU()
        elif teacher_config.activation == 'tanh':
            self.activ = nn.Tanh()
        elif teacher_config.activation is None:
            self.activ = nn.Identity()

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
        if teacher_config.activation == 'relu':
            self.activ = nn.ReLU()
        elif teacher_config.activation == 'tanh':
            self.activ = nn.Tanh()
        elif teacher_config.activation is None:
            self.activ = nn.Identity()

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
}



def train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val):
    torch.manual_seed(teacher_config.model_seed)
    model = models[teacher_config.model_name](teacher_config,  data_config)
    

    criterion = nn.CrossEntropyLoss()

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
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        # Compute accuracy
        with torch.no_grad():
            train_pred = torch.argmax(output, dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_pred = torch.argmax(val_output, dim=1)
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

    return model, {
        'teacher_train_losses': train_losses,
        'teacher_train_accuracies': train_accuracies,
        'teacher_val_accuracies': val_accuracies,
        'teacher_train_loss': train_losses[-1],
        'teacher_train_acc': train_accuracies[-1],
        'teacher_val_acc': val_accuracies[-1],
        'teacher_weight_norms': named_weight_and_bias_norms(model)
    }
    
def run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val, wandb_log=True):

    with torch.no_grad():
        soft_labels = torch.softmax(teacher_model(X_train) / student_config.temperature, dim=-1)

    if student_config.soft_label_treatment is None:
        pass
    elif student_config.soft_label_treatment[0] == 'cut_tail':
        tail_frac_to_cut = student_config.soft_label_treatment[1]
        
        _, indices = torch.sort(soft_labels, descending=True)
        mask = indices < int(data_config.num_labels * tail_frac_to_cut)
        soft_labels[mask] = 0   
        
    elif student_config.soft_label_treatment[0] == 'remove_logits':
        labels = student_config.soft_label_treatment[1]
        for label in labels:
            soft_labels[y_train != label, label] = 0.0

    else:
        raise ValueError(f'Unknown soft label treatment: {student_config.soft_label_treatment}')
        
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

    X_transfer_train = torch.cat(train_X)
    y_transfer_train = torch.cat(train_y)
    X_transfer_test = torch.cat(test_X)
    y_transfer_test = torch.cat(test_y)
    soft_labels_train = torch.cat(soft_labels_train)
    soft_labels_test = torch.cat(soft_labels_test)


    if student_config.soft_label_filtering is None:
        pass
    elif student_config.soft_label_filtering[0] == 'remove_labels':
        labels = student_config.soft_label_filtering[1]
        for label in labels:
            # remove one label from the training set
            mask = y_transfer_train == label
            X_transfer_train = X_transfer_train[~mask]
            soft_labels_train = soft_labels_train[~mask]
            y_transfer_train = y_transfer_train[~mask]
            
    student_model = models[student_config.model_name](student_config, data_config)
    if student_config.weight_decay > 0:
        new_optimizer = optim.Adam(student_model.parameters(), lr=student_config.learning_rate, weight_decay=student_config.weight_decay)
    else:
        new_optimizer = optim.Adam(student_model.parameters(), lr=student_config.learning_rate)
    new_criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    test_accuracies_orig = []
    test_accuracies_zeros = []
    teacher_match_train_acc = []
    teacher_match_test_acc = []

    student_model.to(device)
    X_transfer_train = X_transfer_train.to(device)
    soft_labels_train = soft_labels_train.to(device)
    X_transfer_test = X_transfer_test.to(device)
    soft_labels_test = soft_labels_test.to(device)
    y_transfer_test = y_transfer_test.to(device)
    y_transfer_train = y_transfer_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    for epoch in range(student_config.num_epochs):
        new_optimizer.zero_grad()
        
        new_output = student_model(X_transfer_train)
        loss = new_criterion(new_output, soft_labels_train)
        loss.backward()
        new_optimizer.step()

        with torch.no_grad():
            train_pred = torch.argmax(new_output, dim=1)
            train_acc = (train_pred == y_transfer_train).float().mean().item()
            
            match_teacher_train_acc = (train_pred == soft_labels_train.argmax(dim=-1)).float().mean().item()

            test_output = student_model(X_transfer_test)
            test_loss = new_criterion(test_output, soft_labels_test)
            test_pred = torch.argmax(test_output, dim=1)
            test_acc = (test_pred == y_transfer_test).float().mean().item()
            match_teacher_test_acc = (test_pred == soft_labels_test.argmax(dim=-1)).float().mean().item()
            
            # also compare to original test loss
            test_output_orig = student_model(X_val)
            test_loss_orig = new_criterion(test_output_orig, y_val)
            test_pred_orig = torch.argmax(test_output_orig, dim=1)
            test_acc_orig = (test_pred_orig == y_val).float().mean().item()
            
            # check the test loss specifically on 0 labels
            zeros_mask = y_transfer_test == 0
            test_acc_zeros = (test_pred[zeros_mask] == 0).float().mean().item()
            
        train_losses.append(loss.item())
        train_accuracies.append(train_acc)
        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc)
        test_accuracies_orig.append(test_acc_orig)
        test_accuracies_zeros.append(test_acc_zeros)
        teacher_match_train_acc.append(match_teacher_train_acc)
        teacher_match_test_acc.append(match_teacher_test_acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, '
                f'Test Loss: {test_loss.item():.4f}, Test Acc: {test_acc:.4f}, Test Acc Orig: {test_acc_orig:.4f}, Test Acc Zeros: {test_acc_zeros:.4f}')   
            if wandb_log:
                wandb.log({'train_loss': loss.item(), 'train_acc': train_acc, 'test_loss': test_loss.item(), 'test_acc': test_acc, 'test_acc_orig': test_acc_orig, 'test_acc_zeros': test_acc_zeros})
                
                wandb.log({
                    'student_train_loss': loss.item(),
                    'student_train_acc': train_acc,
                    'student_test_loss': test_loss.item(),
                    'student_match_teacher_train_acc': match_teacher_train_acc,
                    'student_match_teacher_test_acc': match_teacher_test_acc,
                    'student_test_acc': test_acc,
                    'student_test_acc_orig': test_acc_orig,
                    'student_test_acc_zeros': test_acc_zeros,
                    **named_weight_and_bias_norms(student_model,prefix='student_')
                })
    return student_model, {
        'train_losses': (train_losses),
        'train_accuracies': (train_accuracies),
        'test_losses': (test_losses),
        'test_accuracies': (test_accuracies),
        'test_accuracies_orig': (test_accuracies_orig)}
    
    
def combine_dataclasses(data_config, teacher_config, student_config, extra_data):
    
    new_data_config = { 'data_' + key : item  for key, item in dataclasses.asdict(data_config).items()}
    new_teacher_config = { 'teacher_' + key : item  for key, item in dataclasses.asdict(teacher_config).items()}
    new_student_config = { 'student_' + key : item  for key, item in dataclasses.asdict(student_config).items()}
    
    return {**new_data_config, **new_teacher_config, **new_student_config, **extra_data}


if __name__ == "__main__":

    project_name = 'iris_test'

    for _ in range(20):

        for num_samples_per_class in [1000]:
            data_config = DataConfig(num_samples_per_class=num_samples_per_class)
            X_train, y_train, X_val, y_val = create_dataset(data_config)
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            for hidden_dim_teacher in [500]:
                
                teacher_config = TeacherConfig(model_name='one_layer_network',hidden_dim=hidden_dim_teacher, num_epochs=500,bias=False)
                teacher_model, teacher_results = train_teacher(teacher_config, data_config, X_train, y_train, X_val, y_val)

                for hidden_dim in [500]: 
                    for train_frac in [0.7]: 
                        for temperature in [20.0]:
                            student_config = StudentConfig(train_frac=train_frac, temperature=temperature, hidden_dim=hidden_dim, model_name='one_layer_network',num_epochs=10000,bias=False)
                            wandb.init(project=project_name, config=combine_dataclasses(data_config, teacher_config, student_config, teacher_results))
                            student_model, student_results = run_student_exp(data_config, teacher_model, student_config, X_train, y_train, X_val, y_val)
                            wandb.finish()


