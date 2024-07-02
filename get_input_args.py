import argparse

def get_input_args(script_type):
    parser = argparse.ArgumentParser()
    # train.py
    if script_type == 'train':
        parser.add_argument('data_directory', action='store', help='data directory')
        parser.add_argument('--save_dir', type=str, default='', help='path to the folder to save checkpoints')
        parser.add_argument('--arch', type=str, default='vgg13', help='CNN model architecture to use')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
        parser.add_argument('--hidden_units', type=int, default=512, help='hidden units')
        parser.add_argument('--epochs', type=int, default=1, help='epochs')
        parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    # predict.py
    elif script_type == 'predict':
        parser.add_argument('input_image', action='store', help='full path to input image')
        parser.add_argument('checkpoint', action='store', help='src checkpoint')
        parser.add_argument('--top_k', type=int, default=3, help='top k most likely classes')
        parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='file mapping categories to real names')
        parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    
    return parser.parse_args()