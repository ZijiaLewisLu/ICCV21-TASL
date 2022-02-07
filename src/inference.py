#!/usr/bin/python3.7

import numpy as np
import multiprocessing as mp
from queue import Empty
from .utils.network import Forwarder, Net
from .utils.grammar import PathGrammar, SingleTranscriptGrammar
from .utils.viterbi import Viterbi
import os
import time
import json
from pprint import pprint

### helper function for parallelized Viterbi decoding ##########################
def read_file(filename, dataset, index2label):

    # read ground truth
    video = os.path.basename(filename)
    if video.endswith("_action_alignment"):
        video = video[:len(video)-len("_action_alignment")]
    s, t, ground_truth = dataset[video]
    ground_truth = list(map(index2label.get, ground_truth))

    # read recognized sequence
    with open(filename, 'r') as f:
        recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
        f.close()

    ground_truth = np.array(ground_truth)
    recognized = np.array(recognized)
    return ground_truth, recognized

def compute_mof(gt_list, pred_list):
    match, total = 0, 0
    for gt_label, recognized in zip(gt_list, pred_list):

        correct = recognized == gt_label

        match += correct.sum()
        total += len(gt_label)

    mof = match / total
    return mof 

def compute_IoU_IoD(gt_list, pred_list):

    IOU, IOU_NB = [], []
    IOD, IOD_NB = [], []
    for ground_truth, recognized in zip(gt_list, pred_list):

        unique = list(set(np.unique(ground_truth))) #.union(set(np.unique(recognized))) 

        video_iou = []
        video_iod = []
        for i in unique:
            recog_mask = recognized == i
            gt_mask = ground_truth == i
            union = np.logical_or(recog_mask, gt_mask).sum()
            intersect = np.logical_and(recog_mask, gt_mask).sum() # num of correct prediction
            num_recog = recog_mask.sum()
            
            video_iou.append(intersect / (union+ 1e-6))
            video_iod.append(intersect / (num_recog + 1e-6))

        
        IOU.append(np.mean(video_iou))
        IOD.append(np.mean(video_iod))

        video_iou_noBG = [ v for (a, v) in zip(unique, video_iou) if a != 0 ]
        IOU_NB.append(np.mean(video_iou_noBG))

        video_iod_noBG = [ v for (a, v) in zip(unique, video_iod) if a != 0 ]
        IOD_NB.append(np.mean(video_iod_noBG))

        
    return np.mean(IOU), np.mean(IOU_NB), np.mean(IOD), np.mean(IOD_NB)


def compute_score(filelist, dataset, index2label):
    gt_list = []
    pred_list = []
    for filename in filelist:
        gt, pred = read_file(filename, dataset, index2label)
        gt_list.append(gt)
        pred_list.append(pred)

    mof = compute_mof(gt_list, pred_list)
    iox = compute_IoU_IoD(gt_list, pred_list)
    result = {
        'MoF' : mof,
        'IoU' : iox[0],
        'IoU_noBG': iox[1],
        'IoD' : iox[2],
        'IoD_noBG' : iox[3],
    }

    return result

def forward_videos(net, dataset, window_size=21):
    log_probs = dict()
    recons_error_dict = dict()
    forwarder = Forwarder(net, dataset.n_classes)
    start = time.time()

    for i, data in enumerate(dataset):
        video, sequence, trans, gt = data
        feature, embs, lp, recons_error = forwarder.forward(sequence, window_size=window_size) 
        lp_cali = lp = lp.detach().cpu().numpy()
        if not isinstance(recons_error, list):
            recons_error = recons_error.detach().cpu().numpy()

        log_probs[video] = lp_cali
        recons_error_dict[video] = recons_error

    duration = ( time.time() - start ) / 60
    print("Forward Videos Done - %f m" % duration)
    return log_probs, recons_error_dict

def segment_decode(queue, save_dir, log_probs: dict, decoder: Viterbi, index2label, window, step, overwrite):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            save_fname = os.path.join(save_dir, video)
            if os.path.exists(save_fname) and not overwrite:
                with open(save_fname) as fp:
                    lines = fp.readlines()
                if len(lines) == 7:
                    continue
            else:
                score, labels, segments = decoder.decode( log_probs[video] )
                trancript = [s.label for s in segments]
                score, labels, segments = \
                        decoder.stn_decode( log_probs[video], segments, trancript, window, step)
                # save result
                with open(save_fname, 'w') as f:
                    f.write( '### Recognized sequence: ###\n' )
                    f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                    f.write( '### Score: ###\n' + str(score) + '\n')
                    f.write( '### Frame level recognition: ###\n')
                    f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )

        except Empty:
            pass

def hie_segment_decode(queue, save_dir, log_probs: dict, decoder: Viterbi, index2label, 
                        rep_grammar, cluster_grammar,
                        window, step, overwrite):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            save_fname = os.path.join(save_dir, video)
            if os.path.exists(save_fname) and not overwrite:
                with open(save_fname) as fp:
                    lines = fp.readlines()
                if len(lines) == 7:
                    continue
            else:
                decoder.grammar = rep_grammar
                score, labels, segments = decoder.decode( log_probs[video] )

                rep_string = " ".join([ str(s.label) for s in segments ])
                decoder.grammar = cluster_grammar[rep_string]
                score, labels, segments = decoder.decode( log_probs[video] )

                trancript = [s.label for s in segments]
                score, labels, segments = \
                        decoder.stn_decode( log_probs[video], segments, trancript, window, step)
                # save result
                with open(save_fname, 'w') as f:
                    f.write( '### Recognized sequence: ###\n' )
                    f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                    f.write( '### Score: ###\n' + str(score) + '\n')
                    f.write( '### Frame level recognition: ###\n')
                    f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )

        except Empty:
            pass

def alignment_decode(queue, save_dir, log_probs: dict, transcripts: dict, decoder: Viterbi, index2label, 
                        window, step, overwrite):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            save_fname = os.path.join(save_dir, video + "_action_alignment")
            if os.path.exists(save_fname) and not overwrite:
                with open(save_fname) as fp:
                    lines = fp.readlines()
                if len(lines) == 7:
                    continue
            else:
                decoder.grammar = SingleTranscriptGrammar(transcripts[video], len(index2label))
                score, labels, segments = decoder.decode( log_probs[video] )
                trancript = [s.label for s in segments]
                score, labels, segments = \
                        decoder.stn_decode( log_probs[video], segments, trancript, window, step)

                # save result
                with open(save_fname, 'w') as f:
                    f.write( '### Recognized sequence: ###\n' )
                    f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                    f.write( '### Score: ###\n' + str(score) + '\n')
                    f.write( '### Frame level recognition: ###\n')
                    f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )

        except Empty:
            pass

def action_segmentation(save_dir, dataset, log_probs: dict, viterbi_decoder: Viterbi, grammar: PathGrammar, index2label, 
                            n_threads=32,
                            edge_window=None, edge_step=None,
                            overwrite=False,
                            hie_grammar=None):

    print("Action Segmentation")
    viterbi_decoder.grammar = grammar
    queue = mp.Queue()
    for k in log_probs:
        queue.put(k)

    ### Viterbi decoding
    start = time.time()

    procs = []
    for i in range(n_threads):
        if hie_grammar is None:
            p = mp.Process(target = segment_decode, 
                    args = (queue, save_dir, log_probs, viterbi_decoder, index2label, edge_window, edge_step, overwrite) )
        else:
            rep_grammar, cluster_grammar = hie_grammar
            p = mp.Process(target = hie_segment_decode, 
                    args = (queue, save_dir, log_probs, viterbi_decoder, index2label, 
                                    rep_grammar, cluster_grammar, edge_window, edge_step, overwrite)
                    )

        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    duration = (time.time() - start) / 60
    for p in procs:
        p.terminate()

    ### evaluate
    flist = [ os.path.join(save_dir, video) for video in dataset.videos() ]
    result = compute_score(flist, dataset, index2label)

    # print frame accuracy (1.0 - frame error rate)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Action Segmentation. Time %f m" % duration)
    pprint(result)
    with open(os.path.join(save_dir, 'segment_metrics.json'), 'w') as fp:
        json.dump(result, fp)


def action_alignment(save_dir, dataset, log_probs, viterbi_decoder, index2label, 
                        n_threads=32,
                        edge_window=None, edge_step=None, overwrite=False):

    queue = mp.Queue()
    for k in log_probs:
        queue.put(k)

    ### Viterbi decoding
    start = time.time()

    procs = []
    for i in range(n_threads):
        p = mp.Process(target = alignment_decode,
                args = (queue, save_dir, log_probs, dataset.transcript, viterbi_decoder, 
                            index2label, edge_window, edge_step, overwrite) )
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

    duration = (time.time() - start) / 60
    # print("Time %f m" % duration)
    for p in procs:
        p.terminate()

    ### evaluate
    flist = [ os.path.join(save_dir, video + "_action_alignment" ) for video in dataset.videos() ]
    result = compute_score(flist, dataset, index2label)

    print("-------------------------------Action Alignment. Time %f m" % duration)
    pprint(result)
    with open(os.path.join(save_dir, 'alignment_metrics.json'), 'w') as fp:
        json.dump(result, fp)
