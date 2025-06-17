from pathlib import Path
import wandb
import pandas as pd
from tqdm.notebook import tqdm

FIGURE_DIR = Path("figures/neurips2025")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

wandb_username = "<your_username_for_wandb"

alpha_t_label_color = "#00FF00"
alpha_t_label_name = r'$\alpha^T_{label}$'

alpha_s_id_color = "#0000FF"
alpha_s_id_name = r'$\alpha^S_{id}$'

alpha_s_label_color = "#FF00FF"
alpha_s_label_name = r'$\alpha^S_{label}$'

alpha_s_id_color_up = 'blue'
alpha_s_id_color_low = 'blue'
alpha_s_id_name_up = r'$\alpha^S_{\leq id}$'
alpha_s_id_name_low = r'$\alpha^S_{\geq id}$'

alpha_s_shuffle_label_color = "#FF0000"
alpha_s_shuffle_label_name = r'$\alpha^{S-shuffle}_{label}$'

metric_styles = {
    'student_test_acc': {
        'color': 'tab:orange',
        'linestyle': '-',
        'label': r'$\mathrm{acc}^{\mathrm{S}}_{\mathrm{test}}$',
    },
    'student_train_acc': {
        'color': 'tab:purple',
        'linestyle': '--',
        'label': r'$\mathrm{acc}^{\mathrm{S}}_{\mathrm{train}}$',
    },
    'student_test_acc_orig': {
        'color': 'tab:green',
        'linestyle': '-',
        'label': r'$\mathrm{acc}^{\mathrm{S}}_{\mathrm{val}}$',
    },
    'teacher_train_acc': {
        'color': 'tab:blue',
        'linestyle': '--',
        'label': r'$\mathrm{acc}^{\mathrm{T}}_{\star}$',
    },
    'teacher_val_acc': {
        'color': 'tab:green',
        'linestyle': '--',
        'label': r'$\mathrm{acc}^{\mathrm{T}}_{val}$',
    },
    'student_val_acc': {
        'color': 'tab:green',
        'linestyle': '--',
        'label': r'$\mathrm{acc}^{\mathrm{S}}_{val}$',
    },
    'student_test_acc_zeros': {
        'color': 'tab:green',
        'linestyle': 'dashed',
        'label': r'$\mathrm{acc}^{\mathrm{S}}_{c=1}$',
    },
    'student_shuffle_test_acc': {
        'color': 'tab:red',
        'linestyle': 'dashed',
        'label': r'$\mathrm{acc}^{\mathrm{S-shuffle}}_{test}$',
    },
    'match_teacher_test_acc': {
        'color': 'tab:blue',
        'linestyle': 'dashed',
        'label': r'$\mathrm{acc}^{\mathrm{S}}_{match-T}$',
    },
    'train_mse': {
        'label': r'$\mathrm{mse}(f^*(\mathcal{D}^S_{train}))$',
    },
    'test_mse': {
        'label': r'$\mathrm{mse}(f^*(\mathcal{D}^S_{test}))$',
    }
}

api = wandb.Api()
def load_result_table(project,samples=2000, load_hist=False):
    # Project is specified by <entity/project-name>
    runs = api.runs(project)

    summary_list = []
    for run in tqdm(runs): 

            res = {**run.summary._json_dict,
                **{k: v for k,v in run.config.items()
                },'name':run.name,'entity': run.entity, 'project': run.project, 'state': run._state, 'run_id': run.id}

            if load_hist:
                history = run.history(samples=samples)  # Adjust `samples` for the number of steps
                res['test_accuracy_hist'] = history["test_acc"]
                res['train_accuracy_hist'] = history["train_acc"]
                res['train_loss_hist'] = history["train_loss"]  
                res['test_loss_hist'] = history["test_loss"]
                res['test_accuracy_orig_hist'] = history["test_acc_orig"]
                res['test_accuracy_zeros_hist'] = history["test_acc_zeros"]

            summary_list.append(res)

    df = pd.DataFrame(summary_list)
    return df

def filter(df, constraints):
    for k, v in constraints.items():
        if k in df.columns:
            df = df[df[k] == v]
    return df

def print_df(a):
    for c in a.columns:
        try:
            if len(a[c].unique()) < 10:
                print(c,a[c].unique())
        except:
            pass