"""
Learning Deep Generative Models of Graphs
Paper: https://arxiv.org/pdf/1803.03324.pdf
This implementation works with a minibatch of size larger than 1 for training and 1 for inference.
"""
import argparse
import datetime
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import setup_train as setup
import torch.nn as nn
from torch import autograd
import helpers
import os
import json
import pickle
import numpy as np
from DGL_DGMG.model_batch import DGMG
from datetime import datetime

save_dir = 'lego-DGMG'
try:
    os.mkdir(save_dir)
except FileExistsError:
    pass
now = datetime.now()
now = now.strftime("%d-%m-%Y--%H-%M-%S")
run_dir = os.path.join(save_dir, now)
os.mkdir(run_dir)
os.mkdir(os.path.join(run_dir, 'models'))



def train_model(args):

    evaluator, printer, train_loader, valid_loader, model, optimizer, scheduler = setup.initialize_training(args, batch_size = args.batch_size)
    # Training
    config = args

    # The file to store results
    with open(os.path.join(run_dir, 'results.h5'), 'wb') as f:
        pickle.dump({}, f)

    with open(os.path.join(run_dir, 'config_save.json'), 'w') as f:
        json.dump(vars(args), f)

    # Training
    model.train()
    print('train loop')
    start_epoch = scheduler.state_dict()['last_epoch']
    for epoch in range(start_epoch, config.epochs):
        epoch_time = time.time()
        epoch_loss = 0
        valid_loss = 0
        for batch, data in enumerate(train_loader):
            # data = data.to(device)
            log_prob = model(batch_size = config.batch_size, actions = data)
            loss = log_prob / config.batch_size
            epoch_loss += loss.item()
            optimizer.zero_grad()

            loss.backward()

            if torch.isnan(loss).any():
                raise Exception('nan loss, {}'.format(batch))

            if config.clip_grad:
                clip_grad_norm_(model.parameters(), config.clip_bound)
            
            optimizer.step()

        scheduler.step()
        if config.valid_split > 0:
            with torch.no_grad():
                for batch, data in enumerate(valid_loader):
                    valid_log_prob = model(batch_size = len(data), actions = data)
                    valid_loss += valid_log_prob.item()

        epoch_time = time.time() - epoch_time
        print('epoch_time: ', epoch_time)

        model.eval()
        helpers.save_model(model, epoch, run_dir, optimizer, scheduler = scheduler)
        with torch.no_grad():
            t2 = time.time()
            result_dir = os.path.join(run_dir, 'samples', 'epoch_{:04d}'.format(epoch))
            gin_metrics, lego_metrics = evaluator.evaluate_model(model, num_samples = config.num_samples, dir = result_dir)
            eval_time = time.time() - t2
            print('done eval, time: ', eval_time)
            printer.update(epoch + 1, {'training loss': epoch_loss,
                                        'validation loss': valid_loss, 
                                        'fid': gin_metrics['fid'],
                                        'invalid lego %': lego_metrics['Overall invalid lego build (%)']})
            
            gin_metrics['training loss'] = epoch_loss ;gin_metrics['validation loss'] = valid_loss
            gin_metrics['eval_time'] = eval_time; gin_metrics['epoch_time'] = epoch_time
            save_results(run_dir, lego_metrics, gin_metrics, epoch)

        model.train()

    helpers.save_model(model, epoch, run_dir, optimizer, scheduler = scheduler)




def save_results(run_dir, lego_metrics, gin_metrics, epoch):
        with open(os.path.join(run_dir, 'results.h5'), 'rb') as f:
            prev_results = pickle.load(f)
        for key, val in lego_metrics.items():
            gin_metrics[key] = val
        prev_results[epoch] = gin_metrics
        with open(os.path.join(run_dir, 'results_temp.h5'), 'wb') as f:
            pickle.dump(prev_results, f)
        os.system('mv {}/results_temp.h5 {}/results.h5'.format(run_dir, run_dir))



if __name__ == '__main__':
    args = setup.parse_args()
    train_model(args)    
   
