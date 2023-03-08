import sys
import signal
# sys.path.insert(0, '/home/aib36/.conda/envs/l45-python3.8/lib/python3.8/site-packages') # hack for HPC computing for correct numpy version

import os
import math
import numpy as np
import time
import argparse
import random
import pickle

from torch_scatter import scatter_sum

# import lovely_tensors as lt
import dadaptation
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, IterableDataset

from torch_geometric.loader import DataLoader as geom_DataLoader

from torchdrug.metrics import f1_max

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment

# from atom3d.datasets import LMDBDataset
from lmdb_dataset import LMDBDataset
from pdb_dataset import ProteinDataset

from gvp import GVP_GNN
from protein_graph import AtomGraphBuilder, _element_alphabet

DATASET_PATH = '/Users/antoniaboca/Downloads/split-by-cath-topology/data'
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/")

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)
_amino_acids = lambda x: {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}.get(x, 20)

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)

# TODO: should this dimensions be bigger, given that we have ~500 possible labels?
GO_DEFAULT_V_DIM = (200, 32)
GO_DEFAULT_E_DIM = (64, 1)

GO_LABELS = 489

class GO_GVP(nn.Module):
    def __init__(self, 
                example, 
                dropout, 
                n_h_node_feats = GO_DEFAULT_V_DIM, 
                n_h_edge_feats = GO_DEFAULT_E_DIM, 
                **model_args):
        super().__init__()
        ns, _ = GO_DEFAULT_V_DIM
        self.gvp = GVP_GNN.init_from_example(example, 
                                            n_h_node_feats = n_h_node_feats,
                                            n_h_edge_feats = n_h_edge_feats, 
                                            **model_args)
        # MLP with 2 hidden layers
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(2*ns, 4*ns), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(4*ns, GO_LABELS)
        )
    
    def forward(self, graph):
        out = self.gvp(graph, scatter_mean=False)
        # Perform readout by summing all features in the graph
        # TODO: Should we concatenate features from all layers?
        readout = scatter_sum(out, index=graph.batch, dim=0)
        out = self.dense(readout)
        return out
    

class RES_GVP(nn.Module):
    def __init__(self, example, dropout, **model_args):
        super().__init__()
        ns, _ = _DEFAULT_V_DIM
        self.gvp = GVP_GNN.init_from_example(example, **model_args)
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2*ns, 20)
        )   
    
    def forward(self, graph):
        out = self.gvp(graph, scatter_mean=False)
        out = self.dense(out)
        return out[graph.ca_idx + graph.ptr[:-1]]

MODEL_SELECT = {'gvp': RES_GVP, 'go': GO_GVP }

class ModelWrapper(pl.LightningModule):
    def __init__(self, model_name, lr, example, dropout, **model_args):
        super().__init__()
        # self.model = model_cls(example, device=self.device, **model_args)
        model_cls = MODEL_SELECT[model_name]
        self.model = model_cls(example, dropout, **model_args)
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        # optimiser = optim.Adam(self.parameters(), lr=self.lr)
        optimiser = dadaptation.DAdaptAdam(self.parameters(), lr=1.0)
        return optimiser

    def training_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = graph.label.to(self.device)
        loss = self.loss_fn(out, labels)
        acc = torch.sum(torch.argmax(out, dim=-1) == labels)
        self.log('train_loss', loss)

        return {'loss': loss, 'acc': acc, 'n_graphs': len(labels)}
    
    def training_epoch_end(self, outputs):
        correct_graphs = 0
        total_graphs = 0
        total_loss = 0.0
        for output in outputs:
            correct_graphs += output['acc']
            total_graphs += output['n_graphs']
            total_loss += output['loss']
        acc = 1.0 * correct_graphs / total_graphs
        total_loss /= len(outputs)

        self.log('train_acc_on_epoch_end', acc, sync_dist=True)
        self.log('train_loss_on_epoch_end', total_loss, sync_dist=True)

    def validation_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = graph.label.to(self.device)
        loss = self.loss_fn(out, labels)
        acc = torch.sum(torch.argmax(out, dim=-1) == labels)
        self.log('val_loss', loss, batch_size = len(labels))

        return {'loss': loss, 'acc': acc, 'n_graphs': len(labels)}
    
    def validation_epoch_end(self, outputs):
        correct_graphs = 0
        total_graphs = 0
        total_loss = 0.0
        for output in outputs:
            correct_graphs += output['acc']
            total_graphs += output['n_graphs']
            total_loss += output['loss']
        acc = 1.0 * correct_graphs / total_graphs
        total_loss /= len(outputs)

        self.log('val_acc_on_epoch_end', acc, sync_dist=True)
        self.log('val_loss_on_epoch_end', total_loss, sync_dist=True)
    
    def test_step(self, graph, batch_idx):
        out = self.model(graph)
        labels = graph.label.to(self.device)
        
        loss = self.loss_fn(out, labels)
        acc = torch.sum(torch.argmax(out[0], dim=-1) == labels)
        self.log('test_acc', acc, batch_size=len(labels))
        self.log('test_loss', loss, batch_size=len(labels))

        return {'loss': loss, 'acc': acc, 'n_graphs': len(labels)}
    
    def test_epoch_end(self, outputs):
        correct_graphs = 0
        total_graphs = 0
        total_loss = 0.0
        for output in outputs:
            correct_graphs += output['acc']
            total_graphs += output['n_graphs']
            total_loss += output['loss']

        acc = 1.0 * correct_graphs / total_graphs
        total_loss /= len(outputs)

        self.log('test_acc_on_epoch_end', acc, sync_dist=True)
        self.log('test_loss_on_epoch_end', total_loss, sync_dist=True)
        
        return {'accuracy': acc, 'test_loss': total_loss}


class GOModelWrapper(pl.LightningModule):
    def __init__(self, model_name, label_weight, lr, example, dropout, **model_args):
        super().__init__()
        # self.model = model_cls(example, device=self.device, **model_args)
        model_cls = MODEL_SELECT[model_name]
        self.model = model_cls(example, dropout, **model_args)
        self.lr = lr
        self.loss_fn = nn.BCEWithLogitsLoss(weight=label_weight, reduction='mean')

    def configure_optimizers(self):
        # optimiser = optim.Adam(self.parameters(), lr=self.lr)
        optimiser = dadaptation.DAdaptAdam(self.parameters(), lr=1.0)
        return optimiser

    def training_step(self, graph, batch_idx):
        out = self.model(graph)

        num_graphs = graph.num_graphs
        labels = graph.targets.to(self.device).reshape(num_graphs, -1)
        assert labels.shape == (num_graphs, GO_LABELS)
        loss = self.loss_fn(out, labels)

        self.log('train_loss', loss)

        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        total_loss = 0.0
        for output in outputs:
            total_loss += output['loss']
        total_loss /= len(outputs)

        self.log('train_loss_on_epoch_end', total_loss, sync_dist=True)

    def validation_step(self, graph, batch_idx):
        out = self.model(graph)
        num_graphs = graph.num_graphs
        labels = graph.targets.to(self.device).reshape(num_graphs, -1)
        assert labels.shape == (num_graphs, GO_LABELS)
        loss = self.loss_fn(out, labels)
        self.log('val_loss', loss, batch_size = len(labels))
        return {'loss': loss, 'preds': out, 'targets': labels}
    
    def validation_epoch_end(self, outputs):
        total_loss = 0.0
        preds_list = []
        targets_list = []
        for output in outputs:
            total_loss += output['loss']
            preds_list.append(output['preds'])
            targets_list.append(output['targets'])
        
        if len(outputs) > 0:
            preds = torch.cat(preds_list)
            targets = torch.cat(targets_list)
            f1_max_score = f1_max(preds, targets)
            self.log('val_f1_max_on_epoch_end', f1_max_score, sync_dist=True)
            total_loss /= len(outputs)
        
        self.log('val_loss_on_epoch_end', total_loss, sync_dist=True)
    
    def test_step(self, graph, batch_idx):
        out = self.model(graph)
        num_graphs = graph.num_graphs
        labels = graph.targets.to(self.device).reshape(num_graphs, -1)  
        assert labels.shape == (num_graphs, GO_LABELS)
        loss = self.loss_fn(out, labels)

        return {'loss': loss, 'preds': out, 'targets': labels}
    
    def test_epoch_end(self, outputs):
        total_loss = 0.0
        preds_list = []
        targets_list = []
        for output in outputs:
            total_loss += output['loss']
            preds_list.append(output['preds'])
            targets_list.append(output['targets'])

        preds = torch.cat(preds_list)
        targets = torch.cat(targets_list)
        f1_max_score = f1_max(preds, targets)

        total_loss /= len(outputs)

        self.log('test_f1_max_on_epoch_end', f1_max_score, sync_dist=True)
        self.log('test_loss_on_epoch_end', total_loss, sync_dist=True)
        
        return {'f1_max': f1_max_score, 'test_loss': total_loss}


class GODataset(IterableDataset):
    def __init__(self, dataset, max_len=None, split='train', shuffle=False):
        self.dataset = dataset
        start, stop = 0, len(dataset)
        self.max_len = max_len
        self.shuffle = shuffle
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
        # import ipdb; ipdb.set_trace()
        if split == 'train':
            start, stop = 0, dataset.num_samples[0]
        elif split == 'valid':
            start = dataset.num_samples[0]
            stop = start + dataset.num_samples[1]
        elif split == 'test':
            start = dataset.num_samples[0] + dataset.num_samples[1]
            stop = start + dataset.num_samples[2]
        else:
            raise Exception("Unknown split for dataset.")
        
        self.start, self.stop = start, stop

    def __iter__(self):
        length = self.stop - self.start
        if self.max_len:
            length = min(length, self.max_len)
        indices = list(range(self.start, self.stop + length))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(indices)
        else:  
            per_worker = int(math.ceil(length / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, length)
            gen = self._dataset_generator(indices[iter_start:iter_end])
        return gen
    
    def _dataset_generator(self, indices):
        if self.shuffle:
            random.shuffle(indices)
        for idx in indices:
            item = self.dataset[idx]

            atoms = item['atoms'].df['ATOM']
            targets = item['targets']
            pdb_file = item['atoms'].pdb_path

            atoms = atoms.rename(columns={
                        'x_coord': 'x', 
                        'y_coord':'y', 
                        'z_coord': 'z', 
                        'element_symbol': 'element'})
            

            graph = self.graph_builder(atoms)
            graph.targets = targets
            graph.pdb_file = pdb_file
            yield graph


class RESDataset(IterableDataset):
    def __init__(self, dataset_path, max_len=None, sample_per_item=None, shuffle=False):
        self.dataset = LMDBDataset(dataset_path)
        self.graph_builder = AtomGraphBuilder(_element_alphabet)
        self.shuffle = shuffle
        self.max_len = max_len
        self.sample_per_item = sample_per_item

    def __iter__(self):
        length = len(self.dataset)
        if self.max_len:
            length = min(length, self.max_len)
        indices = list(range(length))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(indices)
        else:  
            per_worker = int(math.ceil(length / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, length)
            gen = self._dataset_generator(indices[iter_start:iter_end])
        return gen

    def _dataset_generator(self, indices):
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            item = self.dataset[idx]

            atoms = item['atoms']
            # Mask the residues one at a time for a single graph
            if self.sample_per_item:
                limit = self.sample_per_item
            else:
                limit = len(item['labels'])
            for sub in item['labels'][:limit].itertuples():
                _, num, aa = sub.subunit.split('_')
                num, aa = int(num), _amino_acids(aa)
                if aa == 20:
                    continue
                assert aa is not None

                my_atoms = atoms.iloc[item['subunit_indices'][sub.Index]].reset_index(drop=True)
                ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]
                if len(ca_idx) != 1: continue
                graph = self.graph_builder(my_atoms)
                graph.label = aa
                graph.ca_idx = int(ca_idx)
                graph.ensemble = item['id']
                yield graph


def train(args):
    pl.seed_everything(42)
    # train_dataloader = geom_DataLoader(RESDataset(os.path.join(args.data_file, 'train'), shuffle=False, 
    #                     max_len=args.max_len, sample_per_item = args.sample_per_item), 
    #                     batch_size=args.batch_size, num_workers=args.data_workers)
    # val_dataloader = geom_DataLoader(RESDataset(os.path.join(args.data_file, 'val'), 
    #                     max_len=args.max_len, sample_per_item = args.sample_per_item), 
    #                     batch_size=args.batch_size, num_workers=args.data_workers)
    # test_dataloader = geom_DataLoader(RESDataset(os.path.join(args.data_file, 'test'), 
    #                     max_len=args.max_len, sample_per_item = args.sample_per_item), 
    #                     batch_size=args.batch_size, num_workers=args.data_workers)
    print('INFO: loading all graphs into memory...')
    raw_dataset = ProteinDataset(args.data_file) 
    train_dataloader = geom_DataLoader(GODataset(raw_dataset, split='train', max_len=args.max_len, shuffle=False), 
                        batch_size=args.batch_size, num_workers=args.data_workers)
                        
    val_dataloader = geom_DataLoader(GODataset(raw_dataset, split='valid', max_len=args.max_len, shuffle=False), 
                        batch_size=args.batch_size, num_workers=args.data_workers)
    
    test_dataloader = geom_DataLoader(GODataset(raw_dataset, split='test', max_len=args.max_len, shuffle=False), 
                        batch_size=args.batch_size, num_workers=args.data_workers)
    
    # Compute class weights 
    
    if os.path.exists('class_weights.pkl'):
        with open('class_weights.pkl','rb') as handle:
            label_weights = pickle.load(handle)
    else:
        print('Compute raw class weights...')
        stop = raw_dataset.num_samples[0]
        target_totals = torch.zeros((GO_LABELS,))
        for idx in range(stop):
            item = raw_dataset[idx]
            print(f'Number of nodes: {len(item["atoms"].df["ATOM"])}')
            target_totals = target_totals + item['targets']
            del item
        label_weights = target_totals / stop
        with open('class_weights.pkl', 'wb') as handle:
            pickle.dump(label_weights, handle)
    
    pl.seed_everything()
    example = next(iter(train_dataloader))
    model = GOModelWrapper(args.model, label_weights, args.lr, example, args.dropout, n_layers=args.n_layers)

    root_dir = os.path.join(CHECKPOINT_PATH, args.model)
    os.makedirs(root_dir, exist_ok=True)

    wandb_logger = WandbLogger(project='l45-team')
    # lt.monkey_patch()
    if args.gpus > 0:
        if args.slurm:
            plugins = [SLURMEnvironment(requeue_signal=signal.SIGHUP)]
        else:
            plugins = None

        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", 
                                        monitor="val_acc_on_epoch_end")],
            log_every_n_steps=1,
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=args.gpus,
            num_nodes = args.num_nodes,
            strategy='ddp',
            logger=wandb_logger,
            plugins=plugins
        ) 
    else:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", 
                                        monitor="val_acc_on_epoch_end")],
            log_every_n_steps=1,
            max_epochs=args.epochs,
            logger=wandb_logger
        )

    print('Start training...')
    start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)
    end = time.time()
    print('TRAINING TIME: {:.4f} (s)'.format(end - start))
    best_model = ModelWrapper(args.model, args.lr, example, n_layers=args.n_layers)
    best_model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    
    test_result = trainer.test(best_model, test_dataloader)
    print(test_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='go', choices=['gvp','go'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--data_file', type=str, default=DATASET_PATH)
    parser.add_argument('--data_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--sample_per_item', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--slurm', action='store_true', help='Whether or not this is a SLURM job.')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
