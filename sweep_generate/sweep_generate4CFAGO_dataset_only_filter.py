import wandb
import sys

wandb.login()

sweep_config = {
    'name': 'sweep4standard_model_filter-3(CFAGO_dataset_only_miAUPR)',
    'method': 'bayes'
}
metric = {
    'name': 'best_mi-aupr_test',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'hops': {
        'values': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    },
    'n_layers': {
        'values': [1, 2, 3, 4, 5]
    },
    'hidden_dim': {
        'values': [128, 256, 512, 1024]
    },
    'lr': {
        'values': [1e-4, 5e-5, 1e-5, 5e-6]
    },
    'weight_decay': {
        'values': [0.0001, 5e-4, 1e-5]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'att_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'n_heads': {
        'values': [4, 8, 16]
    },
    't': {
        'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    },
    'top': {
        'values': [300, 500, 700]
    }
}

parameters_dict.update({
    'dataset': {
        'value': 'bp'
    },
    'batch_size': {
        'value': 256
    },
    'epochs': {
        'value': 1000
    },
    'patience': {
        'value': 30
    },
    'aspect': {
        'value': 'human'
    },
})

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="wys_NAG_CFAGO_dataset_bp")

print(sweep_id)
