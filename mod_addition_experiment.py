
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import einops
from dataclasses import dataclass
import wandb
import dataclasses

import torch.optim as optim
import matplotlib.pyplot as plt

root = Path('./saved_runs')

possible_functions = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'x2+y2' : lambda x, y: x**2 + y**2,
    'x2+xy+y2': lambda x, y: x**2 + x*y + y**2,
    'oddeven' : lambda x, y: x/y if x % 2 == 1 else x - y,
}

@dataclass
class Config():
    
    seed: int = 0 
    device: t.device = t.device("cuda")
    
    lr: float = 1e-3 
    weight_decay: float = 1.0 
    frac_train: float = 0.3
    num_epochs: int = 20000 
    stopping_thresh: int = -1 
    batch_style: str = 'full' 
    same_noise_each_epoch: bool = False # every epoch the same randomness
    
    save_models: bool = False 
    save_every: int = 100 
    
    p: int = 113 
    fn_name: str = 'add' 
    n_ctx: int = 3
    
    model_type: str = 'transformer'
    transfer_train_fraction : float = None
    
    num_layers: int = 1 
    d_model: int = 128 
    d_mlp: int = 4*d_model
    num_heads: int = 4
    act_type: str = 'ReLU'
    use_ln: bool = False 
    
    mlp_hidden_sizes = [200,200]
    mlp_use_bias: bool = True
    
    random_shift_label_by_pm12: bool = False
    random_shift_label_by: list = None # list of ints 
    random_uniform_label: float = None # fraction of importance given to the value
    transfer_learning_from: str = None # wandb run name
    transfer_learning_temp: float = 10.0
    
    training_label_mode: str = 'noiseless' # ['noiseless', 'sample_soft_labels', 'soft_labels','shuffle_labels']
    
    soft_label_weight: float = 1.0
    
    subsample_batch: float = None  # fraction to subsample from the full batch every epoch
    
    
    @property
    def d_vocab(self):
        return self.p+1
    
    @property
    def d_classes(self):
        return self.p

    @property
    def d_head(self):
        return self.d_model // self.num_heads


def generate_train_test(config: Config):
    pairs = generate_pairs(config)
    np.random.RandomState(config.seed).shuffle(pairs)
    div = int(config.frac_train*len(pairs))
    return torch.tensor(pairs[:div],dtype=torch.long).to(config.device), torch.tensor(pairs[div:],dtype=torch.long).to(config.device)

def generate_pairs(config: Config):
    return [(i, j , config.p) for i in range(config.p) for j in range(config.p)]

def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


class Trainer:
    
    @classmethod
    def init_from_wandb(cls, wandb_run_name, project=None):
        """
        Load a run from WandB and return a Trainer object.
        """
        
        api = wandb.Api()
        
        runs = api.runs(project)

        wandb_run = None
        for run in runs:
            if run.name == wandb_run_name:
                wandb_run = run
        if wandb_run is None:
            raise ValueError(f"Run {wandb_run_name} not found.")    

        config = Config(**wandb_run.config)
        world = cls(config=config,project_name=project)
        file_name = f"{wandb_run_name}-model.pth"  
        wandb_run.file(file_name).download(replace=True)

        state_dict = torch.load(file_name)
        world.model.load_state_dict(state_dict)
    
        return world

    def __init__(self,config : Config, project_name=None):

        if project_name is not None:
            wandb.init(project = project_name, config = dataclasses.asdict(config))
            self.run_name = wandb.run.name
        
        self.config = config
        self._fn = possible_functions[config.fn_name]
    
        
        if config.model_type == 'transformer':
            self.model = Transformer(config)
        elif config.model_type == 'mlp':
            self.model = StandaloneMLP(config)
        else:
            raise ValueError(f"Model type {config.model_type} not recognized.")
        
        self.model.to(config.device)
        
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step/10, 1)) 
        
        self.train_input, self.test_input = generate_train_test(config = config)
        
        
        self.test_labels = self.fn(self.test_input[:,0], self.test_input[:,1])
        self.train_labels = self.fn(self.train_input[:,0], self.train_input[:,1])
        
        if self.config.transfer_train_fraction is not None:
            self.train_index = np.random.RandomState(config.seed).choice(len(self.train_input), int(config.transfer_train_fraction*len(self.train_input)), replace=False)
            self.train_mask = np.zeros(len(self.train_input), dtype=bool)
            self.train_mask[self.train_index] = True
            
            self.mem_test_input = self.train_input[~self.train_mask]
            self.mem_test_labels = self.train_labels[~self.train_mask]
            
            self.train_input = self.train_input[self.train_mask]
            self.train_labels = self.train_labels[self.train_mask]
            
            
            
            self.mem_test_accuracies = []
            self.mem_test_losses = []
            
        
        if config.transfer_learning_from is not None:
            self.transfer_model = Trainer.init_from_wandb(config.transfer_learning_from, project=project_name).model
            
            if config.training_label_mode == 'noiseless':
                raise ValueError("Transfer learning from a model requires soft labels.")
        
        torch.manual_seed(self.config.seed+2)
        if self.config.training_label_mode == 'shuffle_labels':
            self.train_labels_shuffled = torch.permute(self.train_labels)
            
        elif self.config.training_label_mode in ['sample_soft_labels','soft_labels']:
            self.soft_train_labels = self.generate_soft_train_labels()
            
        elif self.config.training_label_mode == 'noiseless':
            pass
        else:
            raise ValueError(f"Training label mode {self.config.training_label_mode} not recognized.")
            
            
            
        if self.config.same_noise_each_epoch:
            torch.manual_seed(self.config.seed+1)
            self.train_labels_noisy = torch.multinomial(self.soft_train_labels, 1).squeeze()
            
        
            
        torch.manual_seed(self.config.seed+23) 
        
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
        
        
    def fn(self,x,y):
        return self._fn(x,y) % self.config.p
    
    
    def sample_from_soft_labels(self):
        if self.config.same_noise_each_epoch:
            train_labels_sampled = self.train_labels_noisy
            
        else:
        
            with torch.no_grad():
                train_labels_sampled = torch.multinomial(self.soft_train_labels, 1).squeeze()
        
        return train_labels_sampled
        
        
    def generate_soft_train_labels(self):
        
        with torch.no_grad():
        
            if self.config.random_shift_label_by_pm12:
                soft_targets = F.one_hot(self.train_labels, num_classes=self.config.p).float()
                m = nn.CircularPad1d((2, 2)) 
                soft_targets = nn.functional.conv1d(m(soft_targets).unsqueeze(1), 
                                                    torch.tensor([[[0.025,0.075,0.8,0.075,0.025]]]).to('cuda'))[:,0,:]
            elif self.config.random_uniform_label is not None:
                soft_targets =  F.one_hot(self.train_labels, num_classes=self.config.p).float() * (self.config.random_uniform_label - 1/self.config.p * (1-self.config.random_uniform_label))
                soft_targets += 1/self.config.p * (1-self.config.random_uniform_label)
                
                
            elif self.config.transfer_learning_from is not None:
                soft_targets =  F.softmax(self.transfer_model(self.train_input) / self.config.transfer_learning_temp, dim=-1)
            else:
                soft_targets = F.one_hot(self.train_labels, num_classes=self.config.p).float()
            
            
        return soft_targets
            
        
    def save_epoch(self, epoch, save_to_wandb = True):
        ''' precondition! train loss and test losses have been appended to '''
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_accuracy': self.train_accuracies[-1],
            'test_accuracy': self.test_accuracies[-1],
            'epoch': epoch,
        }
        if save_to_wandb:
            wandb.log(save_dict)
            print("Saved epoch to wandb")
        if self.config.save_models: 
            (root/self.run_name).mkdir(exist_ok=True, parents=True)
            t.save(save_dict,  root/self.run_name/f"{epoch}.pth")
            print(f"Saved model to { root/self.run_name/f'{epoch}.pth'}")
        self.metrics_dictionary[epoch].update(save_dict)

    
    def train_step(self):
        self.optimizer.zero_grad()  
        
        train_input = self.train_input
        
        if self.config.subsample_batch is not None:
            n_orig = len(train_input)
            n_new = int(self.config.subsample_batch * n_orig)
            new_idx = np.random.permutation(range(n_orig))[:n_new]
        else:
            new_idx = slice(None)
            
        train_logits = self.model(train_input)
        
        if self.config.training_label_mode == 'soft_labels':    
        
            loss_fn = nn.CrossEntropyLoss()
            train_loss = self.config.soft_label_weight * loss_fn(train_logits[new_idx], self.soft_train_labels[new_idx]) + (1.0-self.config.soft_label_weight) * cross_entropy_high_precision(train_logits[new_idx], self.train_labels[new_idx])
            
        elif self.config.training_label_mode == 'sample_soft_labels':
            train_loss = cross_entropy_high_precision(train_logits[new_idx], self.sample_from_soft_labels()[new_idx])
            
        elif self.config.training_label_mode == 'noiseless':
            train_loss = cross_entropy_high_precision(train_logits[new_idx], self.train_labels[new_idx])
            
        elif self.config.training_label_mode == 'shuffle_labels':
            train_loss = cross_entropy_high_precision(train_logits[new_idx], self.train_labels_shuffled[new_idx])
        else:
            raise ValueError(f"Training label mode {self.config.training_label_mode} not recognized.")
        
        train_accuracy = (train_logits.argmax(dim=-1) == self.train_labels).float().mean().item()
        
        train_loss.backward()
        
        self.optimizer.step()
        self.scheduler.step()
        
        train_loss = train_loss.item()
        
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
        
        return train_loss, train_accuracy
        
    def update_test_metrics(self):
        test_input = self.test_input
        test_labels = self.test_labels
        
        with torch.no_grad():
            test_logits = self.model(test_input)
            test_loss = cross_entropy_high_precision(test_logits, test_labels).item()
            test_accuracy = (test_logits.argmax(dim=-1) == test_labels).float().mean().item()
            
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            
            if self.config.transfer_train_fraction is not None:
                mem_test_logits = self.model(self.mem_test_input)
                mem_test_loss = cross_entropy_high_precision(mem_test_logits, self.mem_test_labels).item()
                mem_test_accuracy = (mem_test_logits.argmax(dim=-1) == self.mem_test_labels).float().mean().item()
                
                self.mem_test_accuracies.append(mem_test_accuracy)
                self.mem_test_losses.append(mem_test_loss)
                
        if self.config.transfer_train_fraction is not None:
            return test_loss, test_accuracy, mem_test_loss, mem_test_accuracy
        else:
            return test_loss, test_accuracy
            
    def save_epoch(self, epoch, save_to_wandb = True):
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_accuracy': self.train_accuracies[-1],
            'test_accuracy': self.test_accuracies[-1],
            'epoch': epoch,
        }
        
        if self.config.transfer_train_fraction is not None:
            save_dict['mem_test_loss'] = self.mem_test_losses[-1]
            save_dict['mem_test_accuracy'] = self.mem_test_accuracies[-1]
        
        if save_to_wandb:
            wandb.log(save_dict)
            print("Saved epoch to wandb.")
            
    def visualize_output_logits(self):
        pairs = torch.tensor(generate_pairs(self.config), dtype=torch.long).to(self.config.device)
        logits = self.model(pairs)[:,-1]
        
        plt.imshow(logits.detach().cpu().numpy())
        
        
def train_model(config: Config):
    world = Trainer(config = config)
    print(f'Run name {world.run_name}')

    for epoch in range(config.num_epochs):
        world.train_step()
        if epoch % config.save_every == 0:
            world.update_test_metrics()
            world.save_epoch(epoch)
            print(f'Epoch {epoch}, train loss {np.log(world.train_loss[-1]):.4f}, test loss {np.log(world.train_loss[-1]):.4f}')
        
    world.save_epoch(epoch)
    
    
class Embed(nn.Module):
    '''Define network architecture
    '''
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])
    

    
class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        return (x @ self.W_U)
    


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon
    
    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = t.einsum('ihd,bpd->biph', self.W_K, x)
        q = t.einsum('ihd,bpd->biph', self.W_Q, x)
        v = t.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked/np.sqrt(self.d_head), dim=-1)
        z = t.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        assert act_type in ['ReLU', 'GeLU']
        
    def forward(self, x):
        x = t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config):
        '''this function could be augmented to contain more options for creating different architectures'''
        super().__init__()
        self.embed = Embed(d_vocab = config.d_vocab, d_model = config.d_model)
        self.pos_embed = PosEmbed(max_ctx = config.n_ctx, d_model = config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model = config.d_model,
            d_mlp = config.d_mlp,
            d_head = config.d_head,
            num_heads = config.num_heads,
            n_ctx = config.n_ctx,
            act_type = config.act_type,
            model=[self]) for i in range(config.num_layers)])
        self.unembed = Unembed(d_vocab = config.d_classes, d_model = config.d_model)
        self.use_ln = config.use_ln

    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        self.hiddens = [x]
        for block in self.blocks:
            x = block(x)
            self.hiddens.append(x)
        # x = self.ln(x)
        self.outs = x
        y = self.unembed(x)
        return y[:, -1] # here we only do the loss on the last digit
    

class StandaloneMLP(nn.Module):
    def __init__(self,config):
        super(StandaloneMLP, self).__init__()
        self.p = config.p
        self.input_size = (config.n_ctx-1) * config.p
        self.hidden_sizes = config.mlp_hidden_sizes
        self.output_size = config.p
        self.activations = []
        self.activations_from_abs_input = None
        self.layers = nn.ModuleList()
        self.non_linearity = nn.ReLU()
        self.uses_bias = config.mlp_use_bias
        self.alpha = 1
        self.n_ctx = config.n_ctx
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(self.hidden_sizes)+1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=self.uses_bias))
    
    def forward(self, x, keep_activations=False):
        one_hot = torch.nn.functional.one_hot(x[:,:self.n_ctx-1], num_classes=self.p).float()
        x = one_hot.flatten(start_dim=1)
        
        if keep_activations:
            self.activations = []
        for i, layer in enumerate(self.layers):
            x = self.non_linearity(layer(x)) if i<len(self.layers) -1 else layer(x)
            if keep_activations and i<len(self.layers):
                self.activations.append(x)
        
        return x*self.alpha
    
    
class UnsupervisedTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.transformer = Transformer(config).to(config.device)
        
    def forward(self, x1,x2):
        logits1 = self.transformer(x1)
        logits2 = self.transformer(x2)
        return F.cosine_similarity(logits1, logits2)
        
    
    