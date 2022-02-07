#!/usr/bin/python3

import numpy as np
from .utils.dataset import Dataset 
from .utils.network import Net
from .utils.trainer import Trainer 
from .utils.viterbi import Viterbi
from .utils.length_model import PoissonModel
from .utils.prior import Prior
from .utils.grammar import PathGrammar
from .utils.utils import get_dataset_paths, load_action_mapping, neq_load_customized, generate_exp_name
from .utils.utils import Recorder, get_load_iteration, prepare_save_env
from .inference import forward_videos, action_alignment, action_segmentation
from .home import get_project_base
import argparse
import os
import sys
import torch
import json

def create_dataset(args):
    map_fname, dataset_dir, train_split_fname, test_split_fname = get_dataset_paths(args.data, args.split)
    print("load_data_from", dataset_dir)

    ### read label2index mapping and index2label mapping ###########################
    label2index, index2label = load_action_mapping(map_fname)

    ### read training data #########################################################
    print('read data...')
    with open(train_split_fname, 'r') as f:
        video_list = f.read().split('\n')[0:-1]

    dataset = Dataset(dataset_dir, video_list, label2index, shuffle=True)
    print("Number of training data", len(video_list))
    print(dataset)

    with open(test_split_fname, 'r') as f:
        test_video_list = f.read().split('\n')[0:-1]
    test_dataset = Dataset(dataset_dir, test_video_list, label2index, shuffle=False)
    return dataset_dir, label2index, index2label, dataset, test_dataset

def create_decoder(args):
    maxlen = 80 if args.data == "crosstask" else 2000 # by default, limit all actions to 2000 or 80
    bg_limit = 160 if args.data == 'hollywood' else 0

    defaut_model = np.ones(len(label2index), dtype=np.float32) 
    length_model = PoissonModel(defaut_model, 
                                max_length=maxlen,
                                bg_limit=bg_limit)

    prior_model = Prior(len(label2index))

    decoder = Viterbi(None, length_model, prior_model, 
        frame_sampling=args.decode_sample_rate, max_hypotheses = np.inf)
    return prior_model, length_model, decoder

def create_trainer(args, decoder, net, buffer_size, n_classes):
    recorder = Recorder()

    trainer = Trainer(decoder, net, n_classes, buffer_size=buffer_size, 
                    buffered_frame_ratio=args.buffered_frame_ratio, recorder=recorder)
    
    return trainer, recorder

parser = argparse.ArgumentParser()
# model parameter
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--space_size', default=0, type=int)
parser.add_argument('--pred_size', default=0, type=int)
parser.add_argument('--autoencoder_weight', default=0, type=float, help='')
# viterbi parameter
parser.add_argument('--buffered_frame_ratio', default=25, type=int)
parser.add_argument('--edge_window', default=10, type=int, help="")
parser.add_argument('--edge_step',   default=5 , type=int, help="")
parser.add_argument('--window_size', default=21, type=int)
parser.add_argument('--decode_sample_rate', default=30, type=int)
parser.add_argument('--infer_ew', default=None, type=int, help="")
parser.add_argument('--infer_es', default=None, type=int, help="")
parser.add_argument('--hie_grammar', default=None, type=str, help="")
# training parameter
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--epoch', default=10000, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_decay_iter', default=2500, type=int, help='')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--resume', default=None, type=str, help='path of model to resume')
parser.add_argument('--exp', default='tmp', type=str, help='postfix of experiment name')
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--ali_every', default=1000, type=int)
parser.add_argument('--seg_every', default=5000, type=int)
parser.add_argument('--no_random', action='store_true')
parser.add_argument('--n-threads', default=16, type=int) 

if __name__ == '__main__':

    args = parser.parse_args()

    if args.no_random:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    BASE = get_project_base()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    prefix = "Split%d/" % args.split
    expgroup = generate_exp_name(vars(args))

    if len(args.exp) > 0:
        args.exp = prefix + expgroup + "_" + args.exp
    else:
        args.exp = prefix + expgroup
    if args.infer_ew is None:
        args.infer_ew = args.edge_window
    if args.infer_es is None:
        args.infer_es = args.edge_step

    base_log_dir = os.path.join(BASE, "log", args.data)
    logdir, savedir = prepare_save_env(base_log_dir, args.exp, args)

    dataset_dir, label2index, index2label, dataset, test_dataset = create_dataset(args)


    ### generate path grammar for inference ########################################
    paths = set()
    for _, _, transcript, gt_label in dataset:
        paths.add( ' '.join([index2label[index] for index in transcript]) )
    with open(savedir + '/grammar.txt', 'w') as f:
        f.write('\n'.join(paths) + '\n')

    ### load hie-grammar for hie-segmentation ########################################
    if args.hie_grammar is not None:
        
        with open(args.hie_grammar) as fp:
            subset_dict = json.load(fp)
        cluster_grammar = {}
        all_rep_trans = []
        for rep_trans, transcripts in subset_dict.items():
            rep_trans = rep_trans.split(" ")
            rep_trans = [ label2index[t] for t in rep_trans ]
            all_rep_trans.append(rep_trans)
            rep_string = " ".join(map(str, rep_trans))

            idx_transcripts = []
            for trans in transcripts:
                trans = [ label2index[t] for t in trans ] 
                idx_transcripts.append(trans)
            cluster_grammar[rep_string] = PathGrammar(idx_transcripts, label2index)

        rep_grammar = PathGrammar(all_rep_trans, label2index)
        hie_grammar = (rep_grammar, cluster_grammar)
    else:
        hie_grammar = None
        

    ### create net, decoder, trainer #################################################
    net = Net(dataset.input_dimension, args.hidden_size, args.space_size, len(label2index), 
                pred_size=args.pred_size, autoencoder_weight=args.autoencoder_weight)
    print(net)

    prior_model, length_model, decoder = create_decoder(args)

    buffer_size = len(dataset) 
    print("Buffer Size %d/%d" % (buffer_size, len(dataset) ) )
    trainer, recorder = create_trainer(args, decoder, net, buffer_size, len(index2label))

    start_epoch = 0
    if args.resume:
        load_iteration, netfile, prior, length, buffer = get_load_iteration(args.resume, savedir=savedir)
        print(netfile)
        prev_expdir = "/".join(netfile.split("/")[:-2])
        if load_iteration == 0:
            print("No Checkpoint found, Train from Scratch")
        elif (args.resume == "max" and (load_iteration >= args.epoch)): # or os.path.exists(os.path.join(prev_expdir, "FINISH_PROOF")):
            print("Checkpoint exceed maximum epochs. Ending Experiments")
            sys.exit()
        else:
            assert ("Split%d" % args.split in netfile)
            print("Load from %s, Iteraion %d" % (prev_expdir, load_iteration))
            decoder.prior_model.prior = np.loadtxt(prior)
            decoder.length_model.mean_lengths = np.loadtxt(length)
            decoder.length_model.precompute_prob(length_model.mean_lengths, length_model.max_len)
            state_dict = torch.load(netfile)
            neq_load_customized(net, state_dict)
            start_epoch = load_iteration - 1

            print("Load in buffer")
            trainer.buffer.load(buffer, dataset)
            print("Buffer Finish")

            if load_iteration > args.lr_decay_iter:
                args.lr = args.lr * 0.1

    # HACK
    net.eval()


    for i in range(start_epoch, args.epoch):
        score, log_prob, pred_label, feature = \
                trainer.train(dataset, 
                        batch_size=args.batch_size, learning_rate=args.lr, 
                        window_size=args.window_size,
                        edge_window=args.edge_window, edge_step=args.edge_step)

        # print some progress information
        if (i+1) % args.print_every == 0:
            loss = recorder.mean_reset("loss")
            recons_loss = recorder.mean_reset("recons_loss")
            decode_acc = recorder.mean_reset("decode_acc")
                
            print('[%d]\t L: %.3f, L_r: %.4f, dec_acc: %.3f' % (i+1, loss, recons_loss, decode_acc))

        # adjust learning rate after X iterations
        if args.lr_decay_iter != 0 and (i+1) == args.lr_decay_iter:
            args.lr = args.lr * 0.1

        if (i+1) % args.ali_every == 0 or (i+1) % args.seg_every == 0: 
            ### save model
            print('save snapshot ' + str(i+1), args.exp)
            print("dataset: ", args.data)

            network_file = savedir + '/network.iter-' + str(i+1) + '.net'
            length_file = savedir + '/lengths.iter-' + str(i+1) + '.txt'
            prior_file = savedir + '/prior.iter-' + str(i+1) + '.txt'
            trainer.save_model(network_file, length_file, prior_file)
            trainer.buffer.save(savedir + "/buffer.iter-%d.pk"%(i+1))

            ### inference
            print("\nAutomatic Inference")
            test_save_dir = os.path.join(logdir, "save_rslt_%d" % (i+1))
            os.makedirs(test_save_dir, exist_ok=True)

            with torch.no_grad():
                log_probs, recons_error = forward_videos(net, test_dataset, window_size=args.window_size)

            action_alignment(test_save_dir, test_dataset, log_probs, decoder, index2label, 
                            n_threads=args.n_threads, 
                            edge_window=args.infer_ew, edge_step=args.infer_es,
                            overwrite=True)
            
            if (i+1) % args.seg_every == 0:
                grammar = PathGrammar(savedir + '/grammar.txt', label2index)
                action_segmentation(test_save_dir, test_dataset, log_probs, decoder, grammar, index2label, 
                                n_threads=args.n_threads,
                                edge_window=args.infer_ew, edge_step=args.infer_es,
                                hie_grammar=hie_grammar,
                                overwrite=True)



    ### write a finish_proof file to indicate the exp is finished
    finish_proof_fname = os.path.join(logdir, "FINISH_PROOF")
    open(finish_proof_fname, "w").close()

