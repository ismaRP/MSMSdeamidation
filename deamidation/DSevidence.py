import os
import sys
import warnings

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from anndata import AnnData

import re

from Bio import SeqIO, AlignIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Align import AlignInfo

class peptide():

    def __init__(self, id, intensity, start, end,  bef, aft, sequence, ptms,
                 proteins, leading_proteins, leading_razor_protein):

        self.id = id
        self.intensity = intensity
        self.sequence = sequence
        self.length = len(sequence)

        # Start and end position in leading razor protein
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



class protein():

    def __init__(self, id, prot_info):
        """
        Sample protein
        """
        self.id = id
        self.info = prot_info

        self.sequence = record
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


class sample():

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
        sample_sp = self.info.Species
        prot_dict_tmp = self.prot_dict.copy()
        for prot_id, prot in prot_dict_tmp.items():
            prot_sp = prot.info.Species
            if prot_sp not in sample_sp:
                # Remove protein from sample
                self.prot_dict.pop(prot_id)
                # Go through peptides and remove protein
                for evID, pept in self.pept_dict.items():
                    if prot_id == pept.leading_razor_protein:
                        pept.leading_razor_protein == None
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
                # Remove protein from sample
                self.prot_dict.pop(prot_id)
                # Go through peptides and remove protein
                for evID, pept in self.pept_dict.items():
                    if prot_id == pept.leading_razor_protein:
                        pept.leading_razor_protein == None
                        pept.proteins.remove(prot_id)
                        pept.leading_proteins.remove(prot_id)
                    if prot_id in pept.leading_proteins:
                        pept.leading_proteins.remove(prot_id)
                        pept.proteins.remove(prot_id)
                    if prot_id in pept.proteins:
                        pept.proteins.remove(prot_id)
        # Update sample prot_list
        self.update_prot_list()


    def filter_nonassociated_peptides(self, which='razor'):
        """
        Filters out peptides that are not associated to any protein
        """

        if which == 'razor':
            pept_dict_tmp = self.pept_dict.copy()
            for evID, pept in pept_dict_tmp.items():
                if pept.leading_razor_protein is None:
                    self.pept_dict.pop(seq)
            # Update sample pept_list
            self.update_pept_list()

        elif which == 'leading':
            pept_dict_tmp = self.pept_dict.copy()
            for evID, pept in pept_dict_tmp.items():
                if len(pept.leading_proteins) == 0:
                    self.pept_dict.pop(seq)
            # Update sample pept_list
            self.update_pept_list()

        elif which == 'proteins':
            pept_dict_tmp = self.pept_dict.copy()
            for evID, pept in pept_dict_tmp.items():
                if len(pept.proteins) == 0:
                    self.pept_dict.pop(seq)
            # Update sample pept_list
            self.update_pept_list()


    def createPep2Prot(self):
        pep2prot = []
        pep2lprot = []
        pep2lrprot = []

        for evID, pep in self.pept_dict.items():
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

    def pep2prot_w(which='pep2lprot'):

        self.which_w = which
        # Collect peptide intensities
        Int = [pept.intensity for pept in self.pept_dict.values()]
        Int = np.array(Int).reshape(-1,1)
        if which == 'pep2prot':
            P = self.pep2prot
        elif which == 'pep2lprot':
            P = self.pep2lprot

        relP = P / np.sum(P, axis=1).reshape(-1,1)
        # sumP = np.sum(relP, axis=0)

        max_iter = 15
        iter = 0

        while iter < max_iter:

            TotInt = np.dot(relP.T, Int)
            Norm = np.dot(P, TotInt)
            newrelP = (P * TotInt.T)/Norm

            # newrelP = (newrelP / np.sum(newrelP, axis=0)) * sumP
            newrelP = newrelP / np.sum(newrelP, axis=1).reshape(-1,1)

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

        if which == 'unique':
            W = self.pep2lprot
        elif which == 'weights':
            if self.peptide_weights is None:
                self.update_pept_list()
                self.update_prot_list()
                self.pep2prot_w('pep2lprot')
            W = self.peptide_weights
        elif which == 'razor':
            W = self.pep2lrprot


        for i, (evID, pept) in enumerate(self.pept_dict.items()):
            if len(pept.leading_proteins) > 1 and which == 'unique':
                # Skip peptide
                continue
            # Get modification status of each peptide
            modified = [0 if mod[0]==mod[1] else 1 for mod in pept.ptms]
            pept_int = pept.intensity
            w_i = W[i,:]
            prot_sel = w_i > 0
            # w_i = w_i[prot_idx]
            w_i = w_i[prot_sel]
            wint_i = pept_int * w_i
            # Get protein id to which modifications will be added
            prot_ids = self.prot_list[prot_sel]
            for k, mod in enumerate(pept.ptms):
                mod_id = mod[0]+str(mod[-1])
                # Assign to the associated proteins
                for prot_id, intw_ij in zip(prot_ids, wint_i):
                    prot = self.prot_dict[prot_id]
                    if mod_id not in prot.ptms:
                        prot.ptms[mod_id] = [
                            mod[0], mod[1], mod[-1], 0, 0, 0
                        ]
                    prot.ptms[mod_id][modified[k]+4] += intw_ij
                    if modified[k] == 1:
                        prot.ptms[mod_id][1] = mod[1]

class MQrun():

    def __init__(self, prot_seqs, d):
        self.samples = {}
        self.proteins = set()
        self.prot_seqs = prot_seqs
        self.name = d


class MQdata():

    def __init__(self, prot_f, prot_seqs):

        self.mqruns = []
        self.intensities = np.array([])
        self.prot_seqs = prot_seqs
        self.prot_f = prot_f

    def add_mqrun(self, mqrun):
        self.mqruns.append(mqrun)

    def plot_intensities(self, path=None):

        #plot length vs intensity
        empty = np.array([True if len(v)==0 else False for v in self.intensities])
        intensities = self.intensities[~empty]
        lengths = np.array(range(1,61))[~empty]
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

        fig = plt.figure(figsize=(20,10), dpi=300)
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
        ax.scatter(np.arange(1, len(lengths) + 1), medians, c='b', s = 35,
                   marker='.', zorder=10, label='Length median')
        ax.scatter(np.arange(1, len(lengths) + 1), means, c='g', s = 35,
                   marker='.', zorder=10, label='Length mean')
        plt.legend()
        plt.title('Intensity distribution by length')

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    def map_positions(self, gene, fasta=None, alignment=None,
                      alnformat='clustal', aamod='QN'):

        if alignment is None:
            if fasta is None:
                sequences = []
                if self.prot_seqs is not None:
                    # Use the sequences in MQdata
                    for seqid, record in self.prot_seqs.items():
                        if self.prot_f.loc[record.id].protein_name == gene:
                            sequences.append(record)
                else:
                    # Collect sequences from each mqrun
                    mqr_nonfasta = []
                    global_prot_seqs = {}
                    for mqr in self.mqruns:
                        if mqr.prot_seq is None:
                            mqr_nonfasta.append(mqr.name)
                            continue
                        global_prot_seqs.update(mqr.prot_seq)
                    if len(mqr_nonfasta) > 0:
                        w = 'The following MQ runs do not contain a fasta file:\n'
                        w += ','.join(mqr_nonfasta)
                        warning.warn(w)
                    if not global_prot_seqs:
                        errmsg = 'No FASTA found either at MQdata or MQrun level.'
                        errmsg += '\nPlease use the fasta argument to provide it now.'
                        sys.exit(errmsg)
                    for seqid, record in global_prot_seqs.items():
                        if self.prot_f.loc[record.id].protein_name == gene:
                            sequences.append(record)

            if fasta is not None:
                sequences = []
                for record in SeqIO.parse(fasta, "fasta"):
                    if self.prot_f.loc[record.id].protein_name == gene:
                        sequences.append(record)

            # Print fasta file with the records from specified gene
            with open(gene + ".fasta", "w") as output_handle:
                SeqIO.write(sequences, output_handle, "fasta")

            # Generate multiple alignment
            clustalo_cline = ClustalOmegaCommandline(
                'clustalo',
                infile=gene + ".fasta",
                outfmt=alnformat,
                outfile=gene + ".aln")
            clustalo_cline()
            # Readm multiple alignment
            align = AlignIO.read(gene + ".aln", "clustal")

        elif alignment is not None:
            if fasta is not None:
                warnings.warn('Importing alignment directly')
            # Readm multiple alignment
            align = AlignIO.read(alignment, alnformat)

        # align._alphabet = ProteinAlphabet()
        aligninfo = AlignInfo.SummaryInfo(align)
        self.consensus = aligninfo.gap_consensus()

        aas = set([l for l in aamod])
        aaregex = re.compile('[' + aamod + ']')

        # Calc map positions
        map_positions = {}
        for record in align:
            mp = {}
            ungapped = record.seq.ungap(gap='-')
            gapped = record.seq
            for ugp, gp in zip(aaregex.finditer(str(ungapped)), aaregex.finditer(str(gapped))):
                mp[ugp.start()+1] = gp.start()+1
            map_positions[record.id] = mp

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


    def create_pxm_matrix(self, prot_name):

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
        return(anndata)



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
            num = np.nansum(xs*rs, axis=0)
            den = np.nansum(rs, axis=0)
            newX[i,:] = num/den
            newR[i,:] = den
            newobs.append(unique_sample.loc[sample])
        newobs = pd.concat(newobs, axis=1, ignore_index=True, sort=False).T

        anndata = deamidationMatrix(newX, obs=newobs, var=self.var, layers = {'R':newR})
        return(anndata)

    def __getitem__(self, index) -> "AnnData":
        """Returns a sliced view of the object."""
        oidx, vidx = self._normalize_indices(index)
        return deamidationMatrix(self, oidx=oidx, vidx=vidx, asview=True)



# class deamidationMatrix():
#     """
#     Atributes:
#         - D
#         - Ydata
#         - rel_ints
#         - counts
#         - trps_data
#             (prot_name, seq,  pos, corr_pos 'prot_name-seq-pos')
#         - layers. List of matrices aligned to D
#     """
#     def __get_trps_data(self, sampleTripeps):
#         all_tripeps = {}
#         for sample, tripeps in sampleTripeps.items():
#             # print('{}: {}'.format(sample,tripeps.keys()))
#             for trp, values in tripeps.items():
#                 if trp not in all_tripeps:
#                     # [avg.rel_deamid, counts, norm_int, prot_name, pos, corrected_pos]
#                     seq = trp.split('-')[1]
#                     all_tripeps[trp] = (values[3], values[4], seq, values[5], 0)
#         trps_data = []
#         for trip, values in all_tripeps.items():
#             trps_data.append(values + tuple([trip]))
#         trps_data = np.array(trps_data,
#                              dtype=[
#                                 ('prot_id', '<U60'),
#                                 ('prot_name', '<U10'),
#                                 ('tripep', 'U3'),
#                                 ('position', 'i4'),
#                                 ('corr_pos', 'i4'),
#                                 ('trp_chain', '<U20')])
#         trps_data = np.sort(trps_data, 0, order=['tripep', 'prot_name', 'position'])
#         return trps_data
#
#     def __set_matrix(self, sampleTripeps, sampleInfo):
#         D = []
#         rel_ints = []
#         Ydata = {}
#         counts = []
#         for h in self.Ykeys:
#             Ydata[h] = np.array([])
#         for sample, tripeps in sampleTripeps.items():
#             d = []
#             counts_line = []
#             relint_line = []
#             for trp in self.trps_data:
#                 if trp[5] in tripeps:
#                     d.append(tripeps[trp[5]][0])
#                     counts_line.append(tripeps[trp[5]][1])
#                     relint_line.append(tripeps[trp[5]][2])
#                 else:
#                     d.append(np.nan)
#                     counts_line.append(0)
#                     relint_line.append(0)
#             D.append(d)
#             counts.append(counts_line)
#             rel_ints.append(relint_line)
#             for i, h in enumerate(self.Ykeys[1:]):
#                 Ydata[h] = np.append(Ydata[h], sampleInfo[sample][i])
#             Ydata[self.Ykeys[0]] = np.append(Ydata[self.Ykeys[0]], sample)
#         D = np.array(D)
#         rel_ints = np.array(rel_ints)
#         counts = np.array(counts)
#         self.D = D
#         self.Ydata = Ydata
#         self.counts = counts
#         self.rel_ints = rel_ints
#
#     def __init__(self, sampleTripeps=None, sampleInfo=None, header=None,
#                  D=None, Ydata=None, rel_ints=None, trps_data=None, counts=None,
#                  layers=[]):
#         if ((sampleInfo is not None) and (header is not None)
#             and (sampleTripeps is not None)):
#             #
#             self.Ykeys = header
#             self.trps_data = self.__get_trps_data(sampleTripeps)
#             self.__set_matrix(sampleTripeps, sampleInfo)
#             self.simulated_data = False
#             self.layers = []
#         elif ((D is not None) and (Ydata is not None)
#             and (trps_data is not None)):
#             #
#             self.D = D
#             self.Ydata = Ydata
#             self.counts = counts
#             self.rel_ints = rel_ints
#             self.trps_data = trps_data
#             self.layers = layers
#             if (counts is None) and (rel_ints is None):
#                 self.simulated_data = True
#             else:
#                 self.simulated_data = False
#         else:
#             raise ValueError('Provide either a sampleTripep data structure or\
#                               D, Ydata, counts, rel_ints and trps_data')
#
#     def write_csv(self, path, dfile='deamidation_matrix.csv', cfile='counts_matrix.csv'):
#         """
#         Write .csv files of D and counts
#         """
#         Ddata_toprint = np.array(self.D, dtype = '<U42')
#         Ddata_toprint = np.hstack((self.Ydata['SampleName'].reshape(-1,1),
#                                    Ddata_toprint))
#         if not self.simulated_data:
#             counts_toprint = np.array(self.counts, dtype = '<U42')
#             counts_toprint = np.hstack((self.Ydata['SampleName'].reshape(-1,1),
#                                        counts_toprint))
#
#         Ddata_outf = open(path+'/'+dfile, 'w+')
#         Ddata_outf.write(','.join(np.append(['tripep'], self.trps_data['tripep']))+'\n')
#         Ddata_outf.write(','.join(np.append(['position'], self.trps_data['position']))+'\n')
#         Ddata_outf.write(','.join(np.append(['corr pos'], self.trps_data['corr_pos']))+'\n')
#         Ddata_outf.write(','.join(np.append(['prot name'], self.trps_data['prot_name']))+'\n')
#
#         if not self.simulated_data:
#             counts_outf = open(path+'/'+cfile, 'w+')
#             counts_outf.write(','.join(np.append(['tripep'], self.trps_data['tripep']))+'\n')
#             counts_outf.write(','.join(np.append(['position'], self.trps_data['position']))+'\n')
#             counts_outf.write(','.join(np.append(['corr pos'], self.trps_data['corr_pos']))+'\n')
#             counts_outf.write(','.join(np.append(['prot name'], self.trps_data['prot_name']))+'\n')
#
#         if not self.simulated_data:
#             for dline, cline in zip(Ddata_toprint, counts_toprint):
#                 dline = ','.join(dline)
#                 cline = ','.join(cline)
#                 Ddata_outf.write(dline+'\n')
#                 counts_outf.write(cline+'\n')
#             Ddata_outf.close()
#             counts_outf.close()
#         else:
#             for dline in Ddata_toprint:
#                 dline = ','.join(dline)
#                 Ddata_outf.write(dline+'\n')
#             Ddata_outf.close()
#
#     def set_trps_data(self, trps_data):
#         self.trps_data = trps_data
#
#
#     def log_1_d(self, tol):
#         """
#         Transform data to ln(1-d) and store it in a layer
#         """
#         # Convert to nan d > 1-tol
#         L = np.copy(self.D)
#         L[L > 1-tol] = np.nan
#         L = np.log(1-L)
#         self.layers.append(L)
#
#     def filter_by_pwcounts(self, cutoff=5):
#         """
#         Returns a mask matrix indicating which tripeps to keep
#         after going pairwise through them and removing samples with
#         low counts.
#         """
#         if self.simulated_data:
#             raise ValueError('Simulated data does not contain counts info.')
#         num_tripeps = len(self.trps_data)
#         info_matrix = np.zeros((num_tripeps,num_tripeps))
#         for i in range(num_tripeps):
#             for j in range(num_tripeps):
#                 counts_i = self.counts[:,i]
#                 counts_j = self.counts[:,j]
#                 pairs = np.logical_and(counts_i >= cutoff, counts_j >= cutoff)
#                 if np.sum(pairs) > 10:
#                     info_matrix[i,j] = True
#                 else:
#                     info_matrix[i,j] = False
#         return np.sum(info_matrix, axis=0) > 1
#
#
#     def sort_tripeps(self, idx):
#         sorted_D = self.D[:,idx]
#         sorted_trps_data = self.trps_data[idx]
#         sorted_layers = [None]*len(self.layers)
#         for i in range(len(self.layers)):
#             sorted_layers[i] = self.layers[i][:,idx]
#         if not self.simulated_data:
#             sorted_counts = self.counts[:,idx]
#             sorted_rel_ints = self.rel_ints[:,idx]
#         else:
#             sorted_counts = None
#             sorted_rel_ints = None
#         deamid_mat = deamidationMatrix(D=sorted_D,
#                                        Ydata=self.Ydata,
#                                        counts=sorted_counts,
#                                        rel_ints=sorted_rel_ints,
#                                        trps_data=sorted_trps_data,
#                                        layers=sorted_layers)
#         return(deamid_mat)
#
#
#     def filter_tripeps(self, mask):
#         filt_D = self.D[:,mask]
#         filt_trps_data = self.trps_data[mask]
#         filt_layers = [None]*len(self.layers)
#         for i in range(len(self.layers)):
#             filt_layers[i] = self.layers[i][:,mask]
#         if not self.simulated_data:
#             filt_counts = self.counts[:,mask]
#             filt_rel_ints = self.rel_ints[:,mask]
#         else:
#             filt_counts = None
#             filt_rel_ints = None
#         deamid_mat = deamidationMatrix(D=filt_D,
#                                        Ydata=self.Ydata,
#                                        counts=filt_counts,
#                                        rel_ints=filt_rel_ints,
#                                        trps_data=filt_trps_data,
#                                        layers=filt_layers)
#         return(deamid_mat)
#
#     def filter_samples(self, mask):
#         filt_D = self.D[mask,:]
#         filt_Ydata = {}
#         filt_layers = [None]*len(self.layers)
#         for i in range(len(self.layers)):
#             filt_layers[i] = self.layers[i][mask,:]
#         if not self.simulated_data:
#             filt_counts = self.counts[mask,:]
#             filt_rel_ints = self.rel_ints[mask,:]
#         else:
#             filt_counts = None
#             filt_rel_ints = None
#         for k, v in self.Ydata.items():
#             filt_Ydata[k] = v[mask]
#         deamid_mat = deamidationMatrix(D=filt_D,
#                                        Ydata=filt_Ydata,
#                                        counts=filt_counts,
#                                        rel_ints=filt_rel_ints,
#                                        trps_data=self.trps_data,
#                                        layers=filt_layers)
#         return(deamid_mat)
#
#     def merge_by_tripep(self):
#
#         if len(self.layers) > 0:
#             warnings.warn('The layers of transformed data will be removed.')
#
#         groups = self.trps_data['tripep']
#
#         groups_set = np.sort(list(set(groups)))
#         new_D = []
#         new_trps_data = []
#         new_counts = []
#         new_rel_ints = []
#
#         comb_mat = np.nan_to_num(self.D) * self.rel_ints
#         for gr in groups_set:
#             mask = groups == gr
#             num = np.sum(comb_mat[:,mask], axis=1)
#             if not self.simulated_data:
#                 sum_rel_ints = np.sum(self.rel_ints[:,mask], axis=1)
#                 sum_counts = np.sum(self.counts[:,mask], axis=1)
#                 new_rel_ints.append(sum_rel_ints)
#                 new_counts.append(sum_counts)
#             new_D.append(num/sum_rel_ints)
#
#             new_trps_data.append(('NA', 'NA', gr, 0, 0, 'NA-'+ gr + '-0'))
#         new_D = np.array(new_D).T
#         new_trps_data = np.array(new_trps_data,
#                                  dtype=[
#                                     ('prot_id', '<U35'),
#                                     ('prot_name', '<U10'),
#                                     ('tripep', 'U3'),
#                                     ('position', 'i4'),
#                                     ('corr_pos', 'i4'),
#                                     ('string', '<U45')
#                                     ])
#         if not self.simulated_data:
#             new_counts = np.array(new_counts).T
#             new_rel_ints = np.array(new_rel_ints).T
#         else:
#             new_counts = None
#             new_rel_ints = None
#
#         new_deamid_mat = deamidationMatrix(D=new_D,
#                                            Ydata=self.Ydata,
#                                            counts=new_counts,
#                                            rel_ints=new_rel_ints,
#                                            trps_data=new_trps_data)
#         return new_deamid_mat
#
#
#     def merge_by_pos(self, corr):
#
#         if len(self.layers) > 0:
#             warnings.warn('The layers of transformed data will be removed.')
#
#         if corr == True:
#             groups = self.trps_data['corr_pos']
#         else:
#             groups = self.trps_data['position']
#         groups_set = np.sort(list(set(groups)))
#
#         prot_names = self.trps_data['prot_name']
#         prot_set = np.sort(list(set(prot_names)))
#
#         new_D = []
#         new_trps_data = []
#         new_counts = []
#         new_rel_ints = []
#
#         comb_mat = np.nan_to_num(self.D) * self.rel_ints
#         for pr in prot_set:
#             for gr in groups_set:
#                 mask = np.logical_and(prot_names == pr, groups == gr)
#                 if np.sum(mask) == 0:
#                     continue
#                 num = np.sum(comb_mat[:,mask], axis=1)
#                 if not self.simulated_data:
#                     sum_rel_ints = np.sum(self.rel_ints[:,mask], axis=1)
#                     sum_counts = np.sum(self.counts[:,mask], axis=1)
#                     new_rel_ints.append(sum_rel_ints)
#                     new_counts.append(sum_counts)
#                 new_D.append(num/sum_rel_ints)
#
#                 trp = self.trps_data[mask][0]['tripep']
#                 new_trps_data.append((pr, pr, trp, gr, gr,
#                                       pr + '-' + trp + '-' + str(gr)))
#         new_D = np.array(new_D).T
#         new_trps_data = np.array(new_trps_data,
#                                  dtype=[
#                                     ('prot_id', '<U35'),
#                                     ('prot_name', '<U10'),
#                                     ('tripep', 'U3'),
#                                     ('position', 'i4'),
#                                     ('corr_pos', 'i4'),
#                                     ('string', '<U45')
#                                     ])
#         if not self.simulated_data:
#             new_counts = np.array(new_counts).T
#             new_rel_ints = np.array(new_rel_ints).T
#         else:
#             new_counts = None
#             new_rel_ints = None
#
#         new_deamid_mat = deamidationMatrix(D=new_D,
#                                            Ydata=self.Ydata,
#                                            counts=new_counts,
#                                            rel_ints=new_rel_ints,
#                                            trps_data=new_trps_data)
#         return new_deamid_mat
#
#     def influx(self):
#         """
#         Calculate information influx for each tripeptide
#         """
#
#         num_dims = self.D.shape[1]
#         Ddata = self.D
#         Dresp = np.zeros(self.D.shape)
#         Dresp[~np.isnan(Ddata)] = 1
#         influx = []
#         prop_missing = []
#         for j in range(num_dims):
#             num = 0
#             den = 0
#             dresp_j = Dresp[:,j]
#             p = np.sum(1-dresp_j)/len(dresp_j)
#             prop_missing.append(p)
#             for k in range(num_dims):
#                 dresp_k = Dresp[:,k]
#                 If_jk = (1-dresp_j) * dresp_k # element-wise product
#                 num += np.sum(If_jk)
#                 den += np.sum(dresp_k)
#             influx.append(num/den)
#         return np.array(influx), np.array(prop_missing)
#
#
#
#     def correct_pos(self, prot_info):
#         """
#         Correct positions
#         """
#         trps_data = self.trps_data
#         new_trps_data = []
#         for trp in trps_data:
#             chain = prot_info[trp['prot_id']][0]
#             ch1_start = prot_info[trp['prot_id']][1]
#             ch1_end = prot_info[trp['prot_id']][2]
#             if chain =='COL1A' and trp['position'] < ch1_end:
#                 chain = chain + '1'
#             trp['prot_name'] = chain
#             corr_pos = trp['position'] - ch1_start
#             corr_end = ch1_end - ch1_start
#             if corr_pos<0:
#                 corr_pos = 0
#             if corr_pos > corr_end:
#                 corr_pos = 0
#             trp['corr_pos'] = corr_pos
#             new_trps_data.append(trp)
#         new_trps_data = np.array(new_trps_data,
#                              dtype=[
#                                 ('prot_id', '<U35'),
#                                 ('prot_name', '<U10'),
#                                 ('tripep', 'U3'),
#                                 ('position', 'i4'),
#                                 ('corr_pos', 'i4'),
#                                 ('string', '<U45')
#                                 ])
#         self.set_trps_data(new_trps_data)
#
#     def stagger_pos(self, l_period):
#         """
#         Calculate stagger. Positions are assumet to be corrected
#         """
#         trps_data = self.trps_data
#
#         n_period = np.floor(trps_data['corr_pos']/l_period)
#         trps_data['corr_pos'] = trps_data['corr_pos'] - l_period * n_period
#
#         self.set_trps_data(trps_data)
#
#         return n_period.astype('int') + 1
#
#
#     def correct_stagger_pos(self, prot_info, stagger):
#         trps_data = self.trps_data
#         new_trps_data = []
#         for trp in trps_data:
#             chain = prot_info[trp['prot_id']][0]
#             ch1_start = prot_info[trp['prot_id']][1]
#             trp['prot_name'] = chain
#             corr_pos = trp['position'] - ch1_start
#             if corr_pos<0:
#                 corr_pos = 0
#             n_period = np.floor(corr_pos/stagger)
#             corr_pos = corr_pos - stagger * n_period
#             trp['corr_pos'] = corr_pos
#             new_trps_data.append(trp)
#         new_trps_data = np.array(new_trps_data,
#                              dtype=[
#                                 ('prot_id', '<U35'),
#                                 ('prot_name', '<U10'),
#                                 ('tripep', 'U3'),
#                                 ('position', 'i4'),
#                                 ('corr_pos', 'i4'),
#                                 ('string', '<U45')
#                                 ])
#         self.set_trps_data(new_trps_data)
