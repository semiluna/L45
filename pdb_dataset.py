"""Code adapted from https://github.com/DeepGraphLearning/torchdrug/blob/4e76c3091d3779edc4bb4047c9d3eb84d40906b4/torchdrug/datasets/gene_ontology.py"""
import os
import utils
import csv
import glob
import copy
import warnings
import pickle
from pathlib import Path

from tqdm import tqdm


from biopandas.pdb import PandasPdb

import torch
from torch.utils import data as torch_data
from torch.utils.data import Dataset

from torchdrug import data, utils

class ProteinDataset(Dataset):
    """
    A set of proteins with their 3D structures and GO terms. These terms classify proteins 
    into hierarchically related functional classes organized into three ontologies: molecular 
    function (MF), biological process (BP) and cellular component (CC).
    Statistics (test_cutoff=0.95):
        - #Train: 27,496
        - #Valid: 3,053
        - #Test: 2,991
    Parameters:
        path (str): the path to store the dataset
        branch (str, optional): the GO branch
        test_cutoff (float, optional): the test cutoff used to split the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    url = "https://zenodo.org/record/6622158/files/GeneOntology.zip"
    md5 = "376be1f088cd1fe720e1eaafb701b5cb"
    branches = ["MF", "BP", "CC"]
    processed_file = "gene_ontology.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]

    def __init__(self, path, branch="MF", test_cutoff=0.95, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if branch not in self.branches:
            raise ValueError("Unknown branch `%s` for GeneOntology dataset" % branch)
        self.branch = branch
        if test_cutoff not in self.test_cutoffs:
            raise ValueError("Unknown test cutoff `%.2f` for GeneOntology dataset" % test_cutoff)
        self.test_cutoff = test_cutoff

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(utils.extract(zip_file), "GeneOntology")
        pkl_file = os.path.join(path, self.processed_file)
        
        csv_file = os.path.join(path, "nrPDB-GO_test.csv")
        pdb_ids = []
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin, delimiter=",")
            idx = self.test_cutoffs.index(test_cutoff) + 1
            _ = next(reader)
            for line in reader:
                if line[idx] == "0":
                    pdb_ids.append(line[0])

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = []
            for split in ["train", "valid", "test"]:
                split_path = utils.extract(os.path.join(path, "%s.zip" % split))
                pdb_files += sorted(glob.glob(os.path.join(split_path, split, "*.pdb")))
            self.load_pdbs(pdb_files, verbose=verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        if len(pdb_ids) > 0:
            self.filter_pdb(pdb_ids)

        tsv_file = os.path.join(path, "nrPDB-GO_annot.tsv")
        pdb_ids = [os.path.basename(pdb_file).split("_")[0] for pdb_file in self.pdb_files]
        self.load_annotation(tsv_file, pdb_ids)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("valid"), splits.count("test")]

    def filter_pdb(self, pdb_ids):
        pdb_ids = set(pdb_ids)
        pdb_files = []
        data = []
        for pdb_file, protein in zip(self.pdb_files, self.data):
            if os.path.basename(pdb_file).split("_")[0] in pdb_ids:
                continue
            pdb_files.append(pdb_file)
            data.append(protein)

        self.pdb_files = pdb_files
        self.data = data

    def load_annotation(self, tsv_file, pdb_ids):
        idx = self.branches.index(self.branch)
        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for i in range(12):
                _ = next(reader)
                if i == idx * 4 + 1:
                    tasks = _
            task2id = {task: i for i, task in enumerate(tasks)}
            _ = next(reader)
            pos_targets = {}
            for line in reader:
                pdb_id, pos_target = line[0], line[idx + 1] if idx + 1 < len(line) else None
                pos_target = [task2id[t] for t in pos_target.split(",")] if pos_target else []
                pos_target = torch.LongTensor(pos_target)
                pos_targets[pdb_id] = pos_target

        # fake targets to enable the property self.tasks
        self.targets = task2id
        self.pos_targets = []
        for pdb_id in pdb_ids:
            self.pos_targets.append(pos_targets[pdb_id])

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def load_pdbs(self, pdb_files, transform=None, verbose=0, **kwargs):
        """
        Load the dataset from pdb files.
        Parameters:
            pdb_files (list of str): pdb file names
            transform (Callable, optional): protein sequence transformation function
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(pdb_files)
        print(f'Loading {num_sample} files...')

        self.transform = transform
        self.kwargs = kwargs
        self.data = []
        self.pdb_files = []
        self.sequences = []

        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing dataframes from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            protein = PandasPdb().read_pdb(pdb_file)
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            # self.sequences.append(protein.to_sequence() if protein else None)

    def load_pickle(self, pkl_file, transform=None, verbose=0, **kwargs):
        """
        Load the dataset from a pickle file.
        Parameters:
            pkl_file (str): file name
            transform (Callable, optional): protein sequence transformation function
            verbose (int, optional): output verbose level
            **kwargs
        """
        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample = pickle.load(fin)

            self.transform = transform
            self.kwargs = kwargs
            self.pdb_files = []
            self.data = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            for i in indexes:
                pdb_file, protein = pickle.load(fin)
                self.pdb_files.append(pdb_file) 
                self.data.append(protein)

    def save_pickle(self, pkl_file, verbose=0):
        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.data)
            pickle.dump(num_sample, fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                pdb_dir, pdb_name = os.path.split(self.pdb_files[i])
                split = os.path.basename(pdb_dir)
                pdb_file = os.path.join(split, pdb_name)
                pickle.dump((pdb_file, self.data[i]), fout)
    
    def __len__(self):
        print('WARNING: "len" function only works if the entire dataset is loaded into memory')
        return len(self.data)

    def __getitem__(self, index):
        protein = copy.deepcopy(self.data[index])
        item = {"atoms": protein}
        if self.transform:
            item = self.transform(item)
        indices = self.pos_targets[index].unsqueeze(0)
        values = torch.ones(len(self.pos_targets[index]))
        item["targets"] = utils.sparse_coo_tensor(indices, values, (len(self.targets),)).to_dense()
        return item
