import wandb
import sys

wandb.login()

sweep_config = {
    'name': 'sweep4output_excess_model-3(with-test-filter)_hyperparameter_t',
    'method': 'grid'
}
metric = {
    'name': 'best_fmax_test',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    # 'hops': {
    #     'values': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # },
    # 'n_layers': {
    #     'values': [1, 2, 3, 4]
    # },
    't': {
        'values': [0, 0.1, 0.2, 0.4, 0.5, 0.6]
    },
}

parameters_dict.update({
    'dataset': {
        'value': 'cc'
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
    'hops': {
        'value': 6
    },
    'n_layers': {
        'value': 5
    },
    'peak_lr': {
        'value': 0.0001
    },
    'weight_decay': {
        'value': 1e-5
    },
    'n_heads': {
        'value': 8
    },
    'hidden_dim': {
        'value': 1024
    },
    'dropout': {
        'value': 0.5
    },
    'att_dropout': {
        'value': 0.3
    },
})

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="wys_NAG_DeepGraphGO_dataset_cc")

print(sweep_id)
