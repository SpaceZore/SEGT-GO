import wandb
import sys

wandb.login()

sweep_config = {
    'name': 'sweep4standard_model_BCELoss-1(CFAGO_dataset_only_top500)',
    'method': 'bayes'
}
metric = {
    'name': 'best_Fmax_test',
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
}

parameters_dict.update({
    'dataset': {
        'value': 'bp'
    },
    'top': {
        'value': 500
    },
    # 'n_layers': {
    #     'value': 1
    # },
    # 'n_heads': {
    #     'value': 8
    # },
    'batch_size': {
        'value': 256
    },
    'epochs': {
        'value': 1000
    },
    'patience': {
        'value': 30
    },
    # 'dropout': {
    #     'value': 0.5
    # },
    'aspect': {
        'value': 'human'
    },
})

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="wys_NAG_CFAGO_dataset_bp")

print(sweep_id)
