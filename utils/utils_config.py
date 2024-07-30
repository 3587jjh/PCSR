import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if not config.get('psnr_type'):
        config['psnr_type'] = 'y'
    if not config.get('data_norm'):
        config['data_norm']['mean'] = [0.0, 0.0, 0.0]
        config['data_norm']['std'] = [1.0, 1.0, 1.0]
    if not config.get('seed'):
        config['seed'] = None
    save_path = os.path.join('save', config_path.split('/')[-1][:-len('.yaml')])
    config['save_path'] = save_path
    config['resume_path'] = os.path.join(save_path, 'iter_last.pth')
    return config