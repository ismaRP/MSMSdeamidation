import sys
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData

import re

from Bio import SeqIO, AlignIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Align import AlignInfo


class Peptide:

    def __init__(self, pid, pred_mass, charge, intensity, start, end, bef, aft, sequence, ptms,
                 proteins, leading_proteins, leading_razor_protein):
        self.id = pid
        self.intensity = intensity
        self.sequence = sequence
        self.length = len(sequence)
        self.pred_mass = pred_mass
        self.charge = charge
        # Start and end position in leading razor Protein
        self.start_position = start
        self.end_position = end

        self.aabef = bef
        self.aaaft = aft

        # PTMS
        self.ptms = ptms

        # Initialize preotein data
        self.proteins = proteins
        self.proteins_w = np.array([])
        self.leading_proteins = leading_proteins
        self.leading_proteins_w = np.array([])
        self.leading_razor_protein = leading_razor_protein


class Protein:

    def __init__(self, pid, prot_info):
        """
        Sample Protein
        """
        self.id = pid
        self.info = prot_info

        # self.sequence = record
        # self.gapped_sequence = None

        self.ptms = {}
        # PTMs ids in proteins are defined by residue + global position
        # {
        #     'Q11':
        #         [
        #             residue,
        #             modification,
        #             position,
        #             mapped position,
        #             int_unmodified,
        #             int_modified
        #         ],
        #     ...
        # }


class Sample:

    def __init__(self, sample_info):

        self.pept_dict = {}
        self.pept_list = []

        self.prot_dict = {}
        self.prot_list = []

        self.pep2prot = []
        self.pep2lprot = []
        self.pep2lrprot = []
        self.peptide_weights = None
        self.info = sample_info

    def update_prot_list(self):
        self.prot_list = np.array(list(self.prot_dict))

    def update_pept_list(self):
        self.pept_list = np.array(list(self.pept_dict))

    def filter_species_prots(self):
        """
        Filters out proteins that do not match the sample species
        """
        sample_sp = self.info.species
        prot_dict_tmp = self.prot_dict.copy()
        for prot_id, prot in prot_dict_tmp.items():
            prot_sp = prot.info.species
            if prot_sp not in sample_sp:
                # Remove Protein from sample
                self.prot_dict.pop(prot_id)
                # Go through peptides and remove Protein
                for evID, pept in self.pept_dict.items():
                    if prot_id == pept.leading_razor_protein:
                        pept.leading_razor_protein = None
                        pept.proteins.remove(prot_id)
                        pept.leading_proteins.remove(prot_id)
                    if prot_id in pept.leading_proteins:
                        pept.leading_proteins.remove(prot_id)
                        pept.proteins.remove(prot_id)
                    if prot_id in pept.proteins:
                        pept.proteins.remove(prot_id)
        # Update sample prot_list
        self.update_prot_list()

    def filter_prots(self, keep_names):
        """
        Filter proteins with name not in prot_names
        """
        prot_dict_tmp = self.prot_dict.copy()
        for prot_id, prot in prot_dict_tmp.items():
            prot_n = prot.info.protein_name
            if prot_n not in keep_names:
                # Remove Protein from sample
                _ = self.prot_dict.pop(prot_id)
                # Go through peptides and remove Protein
                for evID, pept in self.pept_dict.items():
                    if prot_id == pept.leading_razor_protein:
                        pept.leading_razor_protein = None
                        pept.proteins.remove(prot_id)
                        pept.leading_proteins.remove(prot_id)
                    if prot_id in pept.leading_proteins:
                        pept.leading_proteins.remove(prot_id)
                        pept.proteins.remove(prot_id)
                    if prot_id in pept.proteins:
                        pept.proteins.remove(prot_id)
        # Update sample prot_list
        self.update_prot_list()
        self.update_pept_list()

    def filter_nonassociated_peptides(self, which='razor'):
        """
        Filters out peptides that are not associated to any Protein
        """

        if which == 'razor':
            pept_dict_tmp = self.pept_dict.copy()
            for evID, pept in pept_dict_tmp.items():
                if pept.leading_razor_protein is None:
                    self.pept_dict.pop(evID)
            # Update sample pept_list
            self.update_pept_list()

        elif which == 'leading':
            pept_dict_tmp = self.pept_dict.copy()
            for evID, pept in pept_dict_tmp.items():
                if len(pept.leading_proteins) == 0:
                    self.pept_dict.pop(evID)
            # Update sample pept_list
            self.update_pept_list()

        elif which == 'proteins':
            pept_dict_tmp = self.pept_dict.copy()
            for evID, pept in pept_dict_tmp.items():
                if len(pept.proteins) == 0:
                    self.pept_dict.pop(evID)
            # Update sample pept_list
            self.update_pept_list()

    def createPep2Prot(self):
        pep2prot = []
        pep2lprot = []
        pep2lrprot = []
        self.update_pept_list()
        self.update_prot_list()
        for pept_id in self.pept_list:
            pep = self.pept_dict[pept_id]
            pep2prot.append(
                [1 if prot in pep.proteins else 0
                 for prot in self.prot_list]
            )
            pep2lprot.append(
                [1 if prot in pep.leading_proteins else 0
                 for prot in self.prot_list]
            )
            pep2lrprot.append(
                [1 if prot == pep.leading_razor_protein else 0
                 for prot in self.prot_list]
            )
        self.pep2prot = np.array(pep2prot)
        self.pep2lprot = np.array(pep2lprot)
        self.pep2lrprot = np.array(pep2lrprot)

    def pep2prot_w(self, which='pep2lprot'):

        self.which_w = which
        # Collect Peptide intensities
        Int = [pept.intensity for pept in self.pept_dict.values()]
        Int = np.array(Int).reshape(-1, 1)
        if which == 'pep2prot':
            P = self.pep2prot
        elif which == 'pep2lprot':
            P = self.pep2lprot
        else:
            sys.exit('Please se a peptide to protein assignment object: pept2prot or pept2lprot')

        relP = P / np.sum(P, axis=1).reshape(-1, 1)
        # sumP = np.sum(relP, axis=0)

        max_iter = 15
        iter = 0

        while iter < max_iter:
            TotInt = np.dot(relP.T, Int)
            Norm = np.dot(P, TotInt)
            newrelP = (P * TotInt.T) / Norm

            # newrelP = (newrelP / np.sum(newrelP, axis=0)) * sumP
            newrelP = newrelP / np.sum(newrelP, axis=1).reshape(-1, 1)

            relP = newrelP

            iter += 1
        self.peptide_weights = relP

    def mod2prot(self, which):
        """
        Mode can be:
            unique: only use uniquely leading associated peptides
            razor: use razor proteins
            weights: use leading proteins using association weights
        """
        self.update_pept_list()
        self.update_prot_list()
        if which == 'unique':
            W = self.pep2lprot
        elif which == 'weights':
            if self.peptide_weights is None:
                self.pep2prot_w('pep2lprot')
            W = self.peptide_weights
        elif which == 'razor':
            W = self.pep2lrprot
        else:
            sys.exit('Please se a peptide to protein assignment mode: unique, using weights or razor')

        for i, (evID, pept) in enumerate(self.pept_dict.items()):
            if len(pept.leading_proteins) > 1 and which == 'unique':
                # Skip Peptide
                continue
            # Get modification status of each Peptide
            modified = [0 if mod[0] == mod[1] else 1 for mod in pept.ptms]
            pept_int = pept.intensity
            w_i = W[i, :]
            prot_sel = w_i > 0
            # w_i = w_i[prot_idx]
            w_i = w_i[prot_sel]
            wint_i = pept_int * w_i
            # Get Protein id to which modifications will be added
            prot_ids = self.prot_list[prot_sel]
            for k, mod in enumerate(pept.ptms):
                mod_id = mod[0] + str(mod[-1])
                # Assign to the associated proteins
                for prot_id, intw_ij in zip(prot_ids, wint_i):
                    prot = self.prot_dict[prot_id]
                    if mod_id not in prot.ptms:
                        prot.ptms[mod_id] = [
                            mod[0], mod[1], mod[-1], -1, 0, 0
                        ]
                    prot.ptms[mod_id][modified[k] + 4] += intw_ij
                    if modified[k] == 1:
                        prot.ptms[mod_id][1] = mod[1]


class MQrun:

    def __init__(self, prot_seqs, d):
        self.samples = {}
        self.proteins = set()
        self.prot_seqs = prot_seqs
        self.name = d


class MQdata:

    def __init__(self, prot_f, prot_seqs):

        self.mqruns = []
        self.intensities = np.array([])
        self.prot_seqs = prot_seqs
        self.prot_f = prot_f

    def add_mqrun(self, mqrun):
        self.mqruns.append(mqrun)

    def plot_intensities(self, path=None):

        # plot length vs intensity
        empty = np.array([True if len(v) == 0 else False for v in self.intensities])
        intensities = self.intensities[~empty]
        lengths = np.array(range(1, 61))[~empty]
        # Calculate stats
        flat_int = np.array([i for v in intensities for i in v])
        iqr = stats.iqr(flat_int)
        med = np.median(flat_int)
        std = np.std(flat_int)
        mean = np.mean(flat_int)
        # Normalize
        # intensities = (intensities - mean)/std
        # Calculate mean and median intensity for each length
        medians = np.array([np.median(v) for v in intensities])
        means = np.array([np.mean(v) for v in intensities])
        # New mean and median
        med1 = np.median([i for v in intensities for i in v])
        mean1 = np.mean([i for v in intensities for i in v])

        fig = plt.figure(figsize=(20, 10), dpi=300)
        ax = fig.add_subplot(111)
        parts = ax.violinplot(intensities, widths=0.7, showextrema=False)
        ax.set_xticks(np.arange(1, len(lengths) + 1))
        ax.set_xticklabels(lengths)
        ax.set_xlim(0.25, len(lengths) + 0.75)
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        ax.set_xlabel('Peptide length')
        ax.set_ylabel('Intensity')

        ax.hlines(med1, xmin=0, xmax=50, color='b',
                  linestyles='dashed', label='Global median')
        ax.hlines(mean1, xmin=0, xmax=50, color='g',
                  linestyles='dashed', label='Global mean')
        ax.scatter(np.arange(1, len(lengths) + 1), medians, c='b', s=35,
                   marker='.', zorder=10, label='Length median')
        ax.scatter(np.arange(1, len(lengths) + 1), means, c='g', s=35,
                   marker='.', zorder=10, label='Length mean')
        plt.legend()
        plt.title('Intensity distribution by length')

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def collect_prot_seqs(self, exclude_nonused=True):
        """
        Collect all prot_seqs at MQdata and MQrun level, keeping only proteins in the data
        """
        mqdata_prots = set()
        if exclude_nonused:
            for mqrun in self.mqruns:
                mqdata_prots.update(mqrun.proteins)

        global_prot_seqs = {}
        # Add the sequences in MQdata if any
        if self.prot_seqs is not None:
            for prot_id, record in self.prot_seqs.items():
                if not exclude_nonused:
                    global_prot_seqs[prot_id] = record
                else:
                    if prot_id in mqdata_prots:
                        global_prot_seqs[prot_id] = record
        # Collect sequences from each mqrun
        mqr_nonfasta = []
        for mqr in self.mqruns:
            if mqr.prot_seqs is None:
                mqr_nonfasta.append(mqr.name)
                continue
            for prot_id, record in mqr.prot_seqs.items():
                if not exclude_nonused:
                    global_prot_seqs[prot_id] = record
                else:
                    if prot_id in mqdata_prots:
                        global_prot_seqs[prot_id] = record

        if len(mqr_nonfasta) > 0:
            w = 'The following MQ runs do not contain a fasta file:\n'
            w += ','.join(mqr_nonfasta)
            print(w)
        if not global_prot_seqs:
            errmsg = 'No FASTA found either at MQdata or MQrun level.'
            errmsg += '\nPlease use the fasta argument to provide it now.'
            return
        else:
            self.prot_seqs = global_prot_seqs

    def map_positions(self, gene, fasta=None, alignment=None, alnformat='clustal',
                      aamod='QN', return_map=False, exclude_nonused=True):
        if alignment is None:
            if fasta is None:
                self.collect_prot_seqs(exclude_nonused)
                # Tranform global_prot_seqs to a list of seq records
                sequences = []
                for seqid, record in self.prot_seqs.items():
                    if self.prot_f.loc[record.id].protein_name in gene:
                        sequences.append(record)
            else:
                sequences = []
                for record in SeqIO.parse(fasta, "fasta"):
                    if self.prot_f.loc[record.id].protein_name in gene:
                        sequences.append(record)

            # Print fasta file with the records from specified gene
            with open('_'.join(gene) + ".fasta", "w") as output_handle:
                SeqIO.write(sequences, output_handle, "fasta")

            # Generate multiple alignment
            clustalo_cline = ClustalOmegaCommandline(
                'clustalo',
                infile='_'.join(gene) + ".fasta",
                outfmt=alnformat,
                outfile='_'.join(gene) + ".aln",
                force=True
            )
            clustalo_cline()
            # Readm multiple alignment
            align = AlignIO.read('_'.join(gene) + ".aln", "clustal")

        else:
            if fasta is not None:
                warnings.warn('Importing alignment directly...')
            # Readm multiple alignment
            align = AlignIO.read(alignment, alnformat)

        # align._alphabet = ProteinAlphabet()
        aligninfo = AlignInfo.SummaryInfo(align)
        self.consensus = aligninfo.gap_consensus()

        # Keep the alignment stored at MQdata
        gapped_prot_seqs = {}
        for record in align:
            gapped_prot_seqs[record.id] = record
        self.gapped_prot_seqs = gapped_prot_seqs

        aas = set([l for l in aamod])
        aaregex = re.compile('[' + aamod + ']')

        # Calc map positions
        map_positions = {}
        for record in align:
            mp = {}
            ungapped = record.seq.ungap(gap='-')
            gapped = record.seq
            for ugp, gp in zip(aaregex.finditer(str(ungapped)), aaregex.finditer(str(gapped))):
                mp[ugp.start() + 1] = gp.start() + 1
            map_positions[record.id] = mp
        # print(map_positions['sp|P02453|CO1A1_BOVIN'][1064])
        # Correct positions
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                for prot_id, protein in sample.prot_dict.items():
                    mapped_ptms = {}
                    for mod_key, ptm in protein.ptms.items():
                        if ptm[0] in aas:
                            pos = ptm[2]
                            map_pos = map_positions[prot_id][pos]
                            map_key = ptm[0] + str(map_pos)
                            # Adjust position
                            ptm[3] = map_pos
                            mapped_ptms[map_key] = ptm
                    # Exchange ptms for the new one with mapped positions
                    protein.ptms = mapped_ptms

        if return_map:
            return map_positions

    def filter_species_prots(self):
        # For each sample, keep only proteins whose species are in the sample
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                sample.filter_species_prots()

    def filter_prots(self, keep_names):
        # For each sample, keep only proteins in keep_names
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                sample.filter_prots(keep_names)

    def filter_nonassociated_peptides(self, which='razor'):
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                sample.filter_nonassociated_peptides(which)

    def createPep2Prot(self):
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                sample.createPep2Prot()

    def calculate_pep2prot_w(self, which='pep2prot'):
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                sample.pep2prot_weights(which=which)

    def assign_mod2prot(self, which):
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                sample.mod2prot(which)

    def position_pxm_matrix(self, prot_name):

        mods_idx = []
        mods_data = []

        prot_data = []
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                for prot_id, protein in sample.prot_dict.items():
                    if protein.info['protein_name'] != prot_name:
                        continue
                    sample_info = sample.info.copy()
                    sample_info['protein_id'] = prot_id
                    sample_info['sample_name'] = sample_info.name
                    prot_data.append(sample_info)
                    for mod_id, mod in protein.ptms.items():
                        mods_idx.append(mod_id)
                        mods_data.append(mod[0:3])
        var = pd.DataFrame(mods_data, index=mods_idx, columns=('aa', 'mod', 'pos'))
        var = var.groupby(var.index).agg(
            {'aa': lambda x: x[0],
             'pos': lambda x: x[0],
             'mod': max}
        )
        obs = pd.concat(prot_data, axis=1, ignore_index=True, sort=False).T

        M = []
        R = []
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                for prot_id, protein in sample.prot_dict.items():
                    m = []
                    r = []
                    if protein.info['protein_name'] != prot_name:
                        continue
                    for mod_id in var.index:
                        if mod_id in protein.ptms:
                            tot_int = (protein.ptms[mod_id][3] +
                                       protein.ptms[mod_id][4])
                            rel_mod = protein.ptms[mod_id][4] / tot_int
                            m.append(rel_mod)
                            r.append(tot_int)
                        else:
                            m.append(np.nan)
                            r.append(0)
                    M.append(m)
                    R.append(r)
        M = np.array(M)
        R = np.array(R)
        l = {'R': R}

        anndata = deamidationMatrix(M, obs=obs, var=var, layers=l)
        return (anndata)

    def sequence_pxm_matrix(self, prot_name, hws=1):

        if self.prot_seqs is None:
            # Collect sequences from each mqrun
            mqr_nonfasta = []
            global_prot_seqs = {}
            for mqr in self.mqruns:
                if mqr.prot_seqs is None:
                    mqr_nonfasta.append(mqr.name)
                    continue
                global_prot_seqs.update(mqr.prot_seqs)
            if len(mqr_nonfasta) > 0:
                w = 'The following MQ runs do not contain a fasta file:\n'
                w += ','.join(mqr_nonfasta)
                warnings.warn(w)
            if not global_prot_seqs:
                errmsg = 'No FASTA found either at MQdata or MQrun level.'
                errmsg += '\nPlease use the fasta argument to provide it now.'
                sys.exit(errmsg)
        else:
            global_prot_seq = self.prot_seqs

        mods_data = []

        prot_data = []
        for mqr in self.mqruns:
            for sample_name, sample in mqr.samples.items():
                for prot_id, protein in sample.prot_dict.items():
                    if protein.info['protein_name'] != prot_name:
                        continue
                    sample_info = sample.info.copy()
                    sample_info['protein_id'] = prot_id
                    sample_info['protein_name'] = sample_info.name
                    prot_data.append(sample_info)
                    for mod_id, mod in protein.ptms.items():
                        # Extract seq window
                        map_pos = mod[3]
                        seq_window = str(self.gapped_prot_seqs[prot_id].seq[map_pos-1-hws:map_pos+hws])
                        mods_data.append([prot_id, mod[0], mod[1], mod[2], map_pos, seq_window])
        var = pd.DataFrame(mods_data, columns=('protein_id', 'aa', 'mod', 'pos', 'map_pos', 'seq_window'))
        # var = var.groupby(var.index).agg(
        #     {'aa': lambda x: x[0],
        #      'pos': lambda x: x[0],
        #      'mod': max}
        # )
        obs = pd.concat(prot_data, axis=1, ignore_index=True, sort=False).T
        print(var)




class deamidationMatrix(AnnData):

    def groupby_sample(self):
        grouped = self.obs.drop('protein_id', 1).groupby('sample_name', as_index=True)
        unique_sample = grouped.agg(lambda x: x[0])
        newX = np.zeros((len(grouped), self.shape[1]))
        newR = np.zeros((len(grouped), self.shape[1]))
        newobs = []
        X = self.X
        R = self.layers['R']

        for i, (sample, idx) in enumerate(grouped.indices.items()):
            xs = X[idx, :]
            rs = R[idx, :]
            num = np.nansum(xs * rs, axis=0)
            den = np.nansum(rs, axis=0)
            newX[i, :] = num / den
            newR[i, :] = den
            newobs.append(unique_sample.loc[sample])
        newobs = pd.concat(newobs, axis=1, ignore_index=True, sort=False).T

        anndata = deamidationMatrix(newX, obs=newobs, var=self.var, layers={'R': newR})
        return (anndata)

    def __getitem__(self, index) -> "AnnData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return deamidationMatrix(self, oidx=oidx, vidx=vidx, asview=True)
