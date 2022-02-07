from ..home import get_project_base
import os
import numpy as np
from glob import glob
import json
import sys

BASE = get_project_base()

def get_dataset_paths(data, split=1):
    if data == "breakfast":
        map_fname = BASE + 'dataset/Breakfast/mapping.txt'
        dataset_dir = BASE + 'dataset/Breakfast/'
        train_split_fname = BASE + 'dataset/Breakfast/split%d.train' % split
        test_split_fname = BASE + 'dataset/Breakfast/split%d.test' % split
    elif data == "crosstask":
        assert split == 1
        map_fname = BASE + 'dataset/CrossTask/mapping.txt'
        dataset_dir = BASE + 'dataset/CrossTask/'
        train_split_fname = BASE + 'dataset/CrossTask/split%d.train' % split
        test_split_fname = BASE + 'dataset/CrossTask/split%d.test' % split
    elif data == "hollywood":
        map_fname = BASE + 'dataset/Hollywood/mapping.txt'
        dataset_dir = BASE + 'dataset/Hollywood/'
        train_split_fname = BASE + 'dataset/Hollywood/split%d.train' % split
        test_split_fname = BASE + 'dataset/Hollywood/split%d.test' % split
    
    return map_fname, dataset_dir, train_split_fname, test_split_fname

def load_action_mapping(map_fname):
    label2index = dict()
    index2label = dict()
    with open(map_fname, 'r') as f:
        content = f.read().split('\n')[0:-1]
        for line in content:
            label2index[line.split()[1]] = int(line.split()[0])
            index2label[int(line.split()[0])] = line.split()[1]

    return label2index, index2label

def generate_exp_name(args):
    if not isinstance(args, dict):
        args = vars(args)

    exp = []
    exp.append("ss%d" % args['space_size'])
    if args.get('pred_size', 0) > 0:
        exp.append("p%d" % args['pred_size'])

    if args.get('autoencoder_weight', 0) > 0:
        exp.append( 'aew' + str(args['autoencoder_weight']) )

    if args.get("edge_window", 10) != 10:
        exp.append("EW%d" % args.get("edge_window"))

    if args.get("edge_step", 5) != 5:
        exp.append("ES%d" % args.get("edge_step"))

    if args.get('lr', 0.01) != 0.01:
        exp.append( 'LR%s' % str(args['lr']) )

    if args.get("lr_decay_iter", 2500) >= args.get("epoch", 10000) or args.get("lr_decay_iter", 2500) == 0:
        exp.append("noDecay")

    exp = "_".join(exp)
    return exp


def neq_load_customized(model, pretrained_dict):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    print('\n=======Check Weights Loading======')
    print('Weights not used from pretrained file:')
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            print(k)
    print('---------------------------')
    print('Weights not loaded into new model:')
    for k, v in model_dict.items():
        if k not in pretrained_dict:
            print(k)
    print('===================================\n')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model

class Recorder():

    def __init__(self):
        self.dict = {}

    def get(self, key):
        if key not in self.dict:
            self.dict[key] = list()
        return self.dict[key]

    def append(self, key, value):
        self.get(key).append(value)

    def extend(self, key, value):
        self.get(key).extend(value)

    def reset(self, key):
        self.dict[key] = []

    def mean_reset(self, key):
        mean = np.mean(self.get(key))
        self.reset(key)
        return mean

    def get_reset(self, key):
        val = self.get(key)
        self.reset(key)
        return val

def get_load_iteration(resume=None, savedir=None):
    
    if resume == "max":
        network_ckpts = glob(savedir + "/network.iter-*.net")
        iterations = [ int(os.path.basename(f)[:-4].split("-")[-1]) for f in network_ckpts ]
        if len(iterations) > 0: 
            load_iteration = max(iterations)
        else:
            load_iteration = 0 # no checkpoint to use
    else:
        load_iteration = os.path.basename(resume)
        load_iteration = int(load_iteration.split('.')[1].split('-')[1])
        savedir = os.path.dirname(resume)

    net = os.path.join(savedir, 'network.iter-' + str(load_iteration) + '.net')
    prior = os.path.join(savedir, 'prior.iter-' + str(load_iteration) + '.txt')
    length = os.path.join(savedir, 'lengths.iter-' + str(load_iteration) + '.txt')
    buffer = os.path.join(savedir, 'buffer.iter-%d.pk' % load_iteration)

    return load_iteration, net, prior, length, buffer 

def create_logdir(log_dir, check_exist=True):
    if check_exist and os.path.exists(log_dir):
        print('\nWARNING: log_dir exists %s\n' % log_dir)

    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(log_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    return log_dir, ckpt_dir


def create_rerun_script(fname):
    with open(fname, 'w') as fp:
        cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        fp.write( "cd " + os.getcwd() + '\n' )
        fp.write("PY="+ sys.executable +'\n')

        if cuda_device:
            cuda_prefix = "CUDA_VISIBLE_DEVICES=%s " % cuda_device
        else:
            cuda_prefix = ""

        fp.write("%s$PY %s\n"%(cuda_prefix, " ".join(sys.argv)))


def log_param(info, args):
    info('============')

    if not isinstance(args, dict):
        args = vars(args)

    keys = sorted(args.keys())
    for k in keys:
        info( "%s: %s" % (k, args[k]) )

    info('============')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def prepare_save_env(logdir, exp_name, args=None, check_exist=True):

    logParentDir = os.path.join(logdir, exp_name)
    logDir, ckptDir = create_logdir(logParentDir, check_exist)

    rerun_fname = os.path.join(logDir, "run.sh")
    create_rerun_script(rerun_fname)


    if args:
        log_param(print, args)
        argSaveFile = os.path.join(logDir, 'args.json')
        with open(argSaveFile, 'w') as f:
            if not isinstance(args, dict):
                args = vars(args)
            json.dump(args, f, indent=True)

    return logDir, ckptDir

