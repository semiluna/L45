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
from protein_graph import AtomGraphBuilder, _element_alphabet, ResidueGraphBuilder

from edge_methods import convert_to_edge_method_params_dict

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
GO_DEFAULT_V_DIM = (100, 32)
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
            nn.Linear(model_args['n_layers'] * ns, 2*ns), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            # nn.Linear(2*ns, 4*ns), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
            nn.Linear(2*ns, GO_LABELS)
        )
    
    def forward(self, graph):
        out = self.gvp(graph, scatter_mean=False)
        # Perform readout by summing all features in the graph
        readout = scatter_sum(out, index=graph.batch, dim=0)
        out = self.dense(readout)
        return out
    

MODEL_SELECT = {'go': GO_GVP }

class GOModelWrapper(pl.LightningModule):
    def __init__(self, model_name, label_weight, lr, example, dropout, adapt, **model_args):
        super().__init__()
        # self.model = model_cls(example, device=self.device, **model_args)
        model_cls = MODEL_SELECT[model_name]
        self.model = model_cls(example, dropout, **model_args)
        self.lr = lr
        self.dadapt = adapt
        self.loss_fn = nn.BCEWithLogitsLoss(weight=label_weight, reduction='mean')

    def configure_optimizers(self):
        # optimiser = optim.Adam(self.parameters(), lr=self.lr)
        if self.dadapt:
            optimiser = dadaptation.DAdaptAdam(self.parameters(), lr=1.0)
        else:
            optimiser = optim.Adam(self.parameters(), lr=self.lr)
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
        return {'loss': loss, 'preds': out.detach(), 'targets': labels.detach()}
    
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

        return {'loss': loss, 'preds': out.detach(), 'targets': labels.detach()}
    
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
    def __init__(
            self, dataset, max_len=None, split='train', shuffle=False, edge_method="radius", edge_method_params=dict()
        ):
        self.dataset = dataset
        start, stop = 0, len(dataset)
        self.max_len = max_len
        self.shuffle = shuffle
        # self.graph_builder = AtomGraphBuilder(
        #     _element_alphabet, edge_method=edge_method, edge_method_params=edge_method_params
        # )
        self.graph_builder = ResidueGraphBuilder(
            edge_method=edge_method, edge_method_params=edge_method_params
        )

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
        indices = list(range(self.start, self.start + length))
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
            if (atoms['residue_name'] == 'UNK').any():
                continue

            targets = item['targets']
            pdb_file = item['atoms'].pdb_path

            atoms = atoms.rename(columns={
                        'x_coord': 'x', 
                        'y_coord':'y', 
                        'z_coord': 'z', 
                        'element_symbol': 'element', 
                        'atom_name': 'name', 
                        'residue_name': 'resname'})
            
            try:
                graph = self.graph_builder(atoms)
                graph.targets = targets
                graph.pdb_file = pdb_file
                yield graph
            except Exception:
                with open('protein_log.txt', 'a+') as handle:
                    print(f'Could not build residue graph for {pdb_file}', file=handle)


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
    pl.seed_everything(args.seed)
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
    edge_method_params = convert_to_edge_method_params_dict(args.edge_params)

    train_dataloader = geom_DataLoader(
        GODataset(
            raw_dataset, split='train', max_len=args.max_len, shuffle=False,
            edge_method=args.edge_method, edge_method_params=edge_method_params,
        ),
        batch_size=args.batch_size,
        num_workers=args.data_workers,
    )

    val_dataloader = geom_DataLoader(
        GODataset(
            raw_dataset, split='valid', max_len=args.max_len, shuffle=False,
            edge_method=args.edge_method, edge_method_params=edge_method_params,
        ),
        batch_size=args.batch_size,
        num_workers=args.data_workers,
    )

    test_dataloader = geom_DataLoader(
        GODataset(
            raw_dataset, split='test', max_len=args.max_len, shuffle=False,
            edge_method=args.edge_method, edge_method_params=edge_method_params,
        ),
        batch_size=args.batch_size,
        num_workers=args.data_workers,
    )

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
        label_weights = torch.maximum(torch.ones_like(label_weights), 
                            torch.minimum(torch.ones_like(label_weights) * 10.0, label_weights))
        with open('class_weights.pkl', 'wb') as handle:
            pickle.dump(label_weights, handle)
    
    pl.seed_everything(args.seed)
    example = next(iter(train_dataloader))
    model = GOModelWrapper(args.model, label_weights, args.lr, example, args.dropout, args.dadapt, n_layers=args.n_layers)

    root_dir = os.path.join(CHECKPOINT_PATH, args.model)
    os.makedirs(root_dir, exist_ok=True)

    if args.resume_checkpoint is None:
        wandb_logger = WandbLogger(project='l45-team')
    else:
        wandb_logger = WandbLogger(project='l45-team', id=args.wandb_id, resume='must')
    if args.gpus > 0:
        if args.slurm:
            plugins = [SLURMEnvironment(requeue_signal=signal.SIGHUP)]
        else:
            plugins = None

        torch.set_float32_matmul_precision('high')
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[
                ModelCheckpoint(mode="max", monitor="val_f1_max_on_epoch_end"), 
                ModelCheckpoint(mode="max", monitor="epoch"), # saves last completed epoch     
            ],
            log_every_n_steps=1,
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=args.gpus,
            num_nodes = args.num_nodes,
            strategy='ddp',
            logger=wandb_logger,
            plugins=plugins,
        ) 
    else:
        trainer = pl.Trainer(
            default_root_dir=root_dir,
            callbacks=[
                ModelCheckpoint(mode="max", monitor="val_f1_max_on_epoch_end"), 
                ModelCheckpoint(mode="max", monitor="epoch"), # saves last completed epoch   
            ],
            log_every_n_steps=1,
            max_epochs=args.epochs,
            logger=wandb_logger,
        )

    print('Start training...')
    ckpt_path = args.resume_checkpoint
    start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    end = time.time()
    print('TRAINING TIME: {:.4f} (s)'.format(end - start))
    best_model = GOModelWrapper(args.model, label_weights, args.lr, example, args.dropout, args.dadapt, n_layers=args.n_layers)
    best_model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    
    test_result = trainer.test(best_model, test_dataloader)
    print(test_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='go', choices=['gvp','go'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--data_file', type=str, default=DATASET_PATH)
    parser.add_argument('--data_workers', type=int, default=0)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--sample_per_item', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--dadapt', action='store_true')
    parser.add_argument('--slurm', action='store_true', help='Whether or not this is a SLURM job.')
    parser.add_argument(
        '--edge_method',
        type=str,
        default='radius',
        help='What edge method to use. Options: ["radius" (default), "knn", "random"]. '
             'Multiple methods can be given as a single string, with each method separated '
             'by a "+". For example, "knn + random" would add edges using knn clustering, '
             'followed by random edges.'
    )
    parser.add_argument(
        "--edge_params",
        nargs=3,
        action="append",
        default=[],
        help='Params to pass to the edge generator function. Usage requires passing 3 arguments -- '
             'the edge method, the key and the value. Example: "--edge_params knn k 3" sets the value '
             'of k to 3 for the knn edge generation method.'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--wandb_id', type=str, default=None)

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
