import wandb
import sys

wandb.login()

sweep_config = {
    'name': 'sweep4output_excess_model-1(with-test-filter)',
    'method': 'bayes'
}
metric = {
    'name': 'best_fmax_test',
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
    'peak_lr': {
        'values': [5e-3, 1e-3, 5e-4, 1e-4]
    },
    'weight_decay': {
        'values': [0.0001, 5e-4, 1e-5]
    },
    'n_heads': {
        'values': [4, 8, 16]
    },
    'hidden_dim': {
        'values': [128, 256, 512, 1024]
    },
    'dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    'att_dropout': {
        'values': [0.1, 0.3, 0.5]
    },
    't': {
        'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
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
})

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="wys_NAG_DeepGraphGO_dataset_cc")

print(sweep_id)
