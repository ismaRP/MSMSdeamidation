import os
import re
import warnings

from Bio import SeqIO

import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform

from deamidation.DSevidence import Peptide, Protein, Sample, MQrun, MQdata
from deamidation.accFunctions import readHeader


class EvidenceBatchReader():

    def __init__(self, datapath, prot_f, samples_f, fasta_f=None,
                 ptm=['de'], aa=['QN'], sep='\t',
                 tr=None, int_threshold=0, sf_exp=[],
                 include=None, exclude=None):

        """
        :param datapath: root folder containing the dataset folders with the evidence.txt and peptides.txt and
        optionally a fasta file.
        :type datapath: Str

        :param prot_f: CSV mapping between Protein ids, as they appear in the evidence.txt and fasta and Protein names.
        It may contain chain start to correct PTM positions.
        :type prot_f: Str

        :param samples_f: CSV sample metadata.
        :type samples_f: Str

        :param fasta_f:  global fasta file
        If fasta_f is a str, it's read as a fasta file with all the sequences
        used in all the MQ runs. If it is an empty string or the file cannot be
        found, it will look for a fasta file within each run dataset folder.
        The sequences are then kept either at a MQdata level or MQrun level.
        If it's None, no sequence info is used.
        :type fasta_f: Str

        :param ptm: list of ptms to analyze. Default: ['de']
        :type ptm: List

        :param aa: list of amino acids potentially containing the PTMs in ptm. Both variables must match.
        Default: ['QN']
        :type aa: List

        :param sep: separator pf the evidence.txt and peptides.txt, which must be the same across datasets.
        Default: "\t"
        :type sep: Str

        :param tr: intensity transformation. Use 'qt' for quantile transformation,
        'lognorm' for taking log of the intensity and normalizing by the total, 'log' for just taking logarithms
        :type tr: Str

        :param int_threshold: peptides with intensities below are not considered
        :type int_threshold: Float

        :param sf_exp: set of datasets for which the Sample name is found in the Experiment field
        :type sf_exp: List, Set

        :param include: only selected datasets are analyzed
        :type include: List, Set

        :param exclude: analyze all datasets but the ones set here.
        If both include and exclude are set, only include will be used.
        :type exclude: List, Set
        """

        if datapath[-1] != '/':
            datapath += '/'

        datasets = [name for name in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, name))]
        if include is not None:
            kept_data = [d for d in datasets if d in include]
        elif exclude is not None:
            kept_data = [d for d in datasets if d not in exclude]
        else:
            kept_data = datasets

        ptm = ['(\(' + p + '\))?' for p in ptm]

        aalist = '([ARNDCEQGHILKMFPUSTWYVX_]{1})'
        aa_s = '([' + ''.join(aa) + '])'
        aa_m = ['[' + a + ']' for a in aa]
        # Build regexS to find position in seq
        regexS = '(?=(' + aalist + aa_s + aalist + '))'
        # Build regexM to find position in modseq
        aamod = [a + mod for a, mod in zip(aa_m, ptm)]
        regexM = '(' + '|'.join(aamod) + ')'

        # Compile regex
        self.regexM = re.compile(regexM)
        self.regexS = re.compile(regexS)

        self.datasets = kept_data
        self.paths = [datapath + d for d in self.datasets]
        self.sep = sep
        self.sf_exp = sf_exp

        if isinstance(prot_f, str):
            self.prot_f = pd.read_csv(prot_f, header=0, index_col=0, sep=',')
        elif isinstance(prot_f, pd.DataFrame):
            self.prot_f = prot_f
        self.prot_set = set(self.prot_f.index)

        if isinstance(samples_f, str):
            self.samples_f = pd.read_csv(samples_f, header=0, index_col=0, sep=',')
        elif isinstance(samples_f, pd.DataFrame):
            self.samples_f = samples_f
        self.samples_set = set(self.samples_f.index)

        self.fasta_f = fasta_f

        self.int_threshold = int_threshold

        if tr in {'qt', 'lognorm'}:
            self.per_sample_tr = True
            if tr == 'qt':
                self.tr = tr
                self.tr_f = self.__qt
            elif tr == 'lognorm':
                self.tr = tr
                self.tr_f = self.__log_norm
        elif tr in {'log'}:
            self.per_sample_tr = False
            if tr == 'log':
                self.tr = tr
                self.tr_f = self.__log
        elif tr is None:
            self.tr = None
            self.tr_f = None

    def readBatch(self):

        prot_seqs = self.__read_mqdata_fasta()
        mqdata = MQdata(self.prot_f, prot_seqs)

        intensities = [[] for i in range(60)]
        for d, p in zip(self.datasets, self.paths):
            print('Reading MaxQuant run {} ...'.format(d))
            if d in self.sf_exp:
                mqrun, intensities = self.importEvidence(p, d, intensities,
                                                         sample_field='Experiment')
            else:
                mqrun, intensities = self.importEvidence(p, d, intensities,
                                                         sample_field='Raw file')
            # Add mqrun only if it contains samples
            if mqrun.samples:
                mqdata.add_mqrun(mqrun)
            else:
                print('\t{} not added. All samples excluded'.format(d))
        intensities = np.array([np.array(v) for v in intensities])
        mqdata.intensities = intensities

        return (mqdata)

    def __read_mqdata_fasta(self):
        prot_seqs = None

        if self.fasta_f is not None:
            if self.fasta_f != '':
                try:
                    prot_seqs = {}
                    f = open(self.fasta_f, 'r')
                    for record in SeqIO.parse(f, 'fasta'):
                        prot_seqs[record.id] = record
                except FileNotFoundError:
                    self.fasta_f = ''
                    wmsg = 'File not found! Looking for fasta files within \n'
                    wmsg += 'the datasets folders...'
                    warnings.warn(wmsg)
        return (prot_seqs)

    def __read_peptides(self, folder, sep='\t'):
        """
        Reads peptides.txt file from MaxQuant output into a
        simple dictionary structure:
        {Peptide id: [start_position, end_position, bef, aft]}
        """
        peptides = {}

        infile = open(folder + '/peptides.txt', 'r')
        header = infile.readline()[:-1].split(sep)

        headerPos = readHeader(['peptideID', 'start', 'end',
                                'bef', 'aft'],
                               ['id', 'Start position', 'End position',
                                'Amino acid before', 'Amino acid after'],
                               header)

        for line in infile:
            line = line[:-1]
            line = line.split(sep)
            peptideID = line[headerPos['peptideID']]
            start = line[headerPos['start']]
            end = line[headerPos['end']]
            bef = line[headerPos['bef']]
            aft = line[headerPos['aft']]

            peptides[peptideID] = [start, end, bef, aft]
        infile.close()
        return peptides

    def __read_mqrun_fasta(self, folder):

        prot_seqs = {}
        fasta_files = [folder + '/' + f for f in os.listdir(folder) if f.endswith('.fasta')]

        if len(fasta_files) != 0:
            for f in fasta_files:
                with open(f, 'r') as f:
                    for record in SeqIO.parse(f, 'fasta'):
                        prot_seqs[record.id] = record
        else:
            prot_seqs = None
        return (prot_seqs)

    def __getmod(self, seq, modseq, start=0):
        seq = '_' + seq + '_'
        # PTMs ids in peptides are defined by residue + local position
        # [
        #    [
        #         residue,
        #         modification,
        #         local position,
        #         Protein position
        #     ],
        # ...
        # ]
        ptms = [[s.group(3), m.group(0), s.start(3), s.start(3) + start]
                for s, m in zip(self.regexS.finditer(seq),
                                self.regexM.finditer(modseq))]
        return (ptms)

    def __qt(self, x, **kwargs):
        transformed = quantile_transform(np.reshape(x, (-1, 1)),
                                         axis=0, copy=True,
                                         n_quantiles=len(x),
                                         output_distribution='normal')
        transformed = transformed.flatten()
        transformed = transformed - np.min(transformed)
        return (transformed)

    def __log_norm(self, x, **kwargs):
        return (np.log((x / np.max(x)) + 1))

    def __log(self, x, **kwargs):
        return (np.log(x + 1))

    def __tr_per_sample(self, folder, sample_field, **kwargs):
        infile = open(folder + '/evidence.txt', 'r')
        header = infile.readline()[:-1].split(self.sep)

        headerPos = readHeader(['intensity', 'sample'],
                               ['Intensity', sample_field],
                               header)
        old_sample = None
        sample_start = 0
        intensity_list = []

        for i, line in enumerate(infile):
            line = line[:-1]
            line = line.split(self.sep)
            sample_name = line[headerPos['sample']]
            intensity = line[headerPos['intensity']]
            intensity = float(intensity) if intensity != '' else 0
            if sample_name == old_sample or old_sample is None:
                intensity_list.append(intensity)
                old_sample = sample_name
                continue
            else:

                to_transform = intensity_list[sample_start:i]
                transformed = self.tr_f(to_transform)
                intensity_list[sample_start:i] = transformed
                intensity_list.append(intensity)

                sample_start = i
                old_sample = sample_name
        # Transform last sample
        to_transform = intensity_list[sample_start:i + 1]
        transformed = self.tr_f(to_transform, kwargs)

        intensity_list[sample_start:i + 1] = transformed
        infile.close()
        return (intensity_list)

    def __tr_one_shot(self, folder, sample_field, **kwargs):
        infile = open(folder + '/evidence.txt', 'r')
        header = infile.readline()[:-1].split(self.sep)

        headerPos = readHeader(['intensity'],
                               ['Intensity'],
                               header)

        intensity_list = []
        # Collect intensities
        for line in infile:
            line = line[:-1]
            line = line.split(self.sep)
            intensity = line[headerPos['intensity']]
            intensity = float(intensity) if intensity != '' else 0
            intensity_list.append(intensity)
        intensity_list = np.array(intensity_list)

        if self.tr is not None:
            intensity_list = self.tr_f(intensity_list, **kwargs)
        return (intensity_list)

    def importEvidence(self, folder, d, intensities=[], sample_field='Raw file'):
        """
        Read evidence.txt file from MaxQuant output

        MQdata =
            {sample1: sampleObject,
             ...
            }

        """
        # Read peptides.txt
        peptides = self.__read_peptides(folder)

        # Read fasta
        prot_seqs = None
        if self.fasta_f == '':
            prot_seqs = self.__read_mqrun_fasta(folder)
            if prot_seqs is None:
                print('\t{} does not contain any fasta file'.format(d))

        mqrun = MQrun(prot_seqs, d)

        infile = open(folder + '/evidence.txt', 'r')
        header = infile.readline()[:-1].split(self.sep)

        headerPos = readHeader(
            [
                'seq',
                'modseq',
                'intensity',
                'sample',
                'proteins',
                'leading_prots',
                'leading_razor_prot',
                'mass',
                'charge',
                'peptideID',
                'evidenceID'
            ],
            [
                'Sequence',
                'Modified sequence',
                'Intensity',
                sample_field,
                'Proteins',
                'Leading proteins',
                'Leading razor protein',
                'Mass',
                'Charge',
                'Peptide ID',
                'id'
            ],
            header)

        if self.tr is not None:
            if self.per_sample_tr:
                tr_intensities = self.__tr_per_sample(folder, sample_field)
            else:
                tr_intensities = self.__tr_one_shot(folder, sample_field)
        else:
            tr_intensities = self.__tr_one_shot(folder, sample_field)

        for i, line in enumerate(infile):
            line = line[:-1]
            line = line.split(self.sep)

            raw_int = line[headerPos['intensity']]
            raw_int = float(raw_int) if raw_int != '' else 0
            if raw_int < self.int_threshold:
                continue

            sample_name = line[headerPos['sample']]
            if sample_name not in self.samples_set:
                continue

            proteins = line[headerPos['proteins']].split(';')
            proteins = [p for p in proteins if p in self.prot_set]
            if len(proteins) == 0:
                continue
            leading_prots = line[headerPos['leading_prots']].split(';')
            leading_prots = [p for p in leading_prots if p in self.prot_set]
            if len(leading_prots) == 0:
                continue
            modseq = line[headerPos['modseq']]
            # _GAP(hy)GADGPAGAP(hy)GTP(hy)GPQ(de)GIAGQ(de)R_
            seq = line[headerPos['seq']]
            evidenceID = line[headerPos['evidenceID']]

            # Get intensity from transofrmed list
            intensity = tr_intensities[i]

            peptideID = line[headerPos['peptideID']]
            mass = line[headerPos['mass']]
            charge = line[headerPos['charge']]
            # Get Peptide-in-Protein position info
            start = peptides[peptideID][0]
            start = int(start) - 1 if start != '' else 0

            end = peptides[peptideID][1]
            end = int(end) if end != '' else 0

            bef = peptides[peptideID][2]
            if bef == '-' or bef == '': bef = '_'
            aft = peptides[peptideID][3]
            if aft == '-' or bef == '': aft = '_'

            intensities[len(seq) - 1].append(intensity)

            leading_razor_prot = line[headerPos['leading_razor_prot']]

            tmp_prot = {}
            for prot_id in proteins:
                prot_info = self.prot_f.loc[prot_id]
                prot = Protein(
                    prot_id, prot_info
                )
                tmp_prot[prot_id] = prot

            ptms = self.__getmod(seq, modseq, start)
            pep = Peptide(peptideID, mass, charge, intensity, start, end, bef, aft, seq, ptms,
                          proteins, leading_prots, leading_razor_prot)

            # Fill sample and Protein data
            if sample_name not in mqrun.samples:
                # If new sample, create and add first Peptide
                sample_info = self.samples_f.loc[sample_name]
                mqrun.samples[sample_name] = Sample(sample_info)
                mqrun.samples[sample_name].pept_dict[evidenceID] = pep
            else:
                # Else (sample already created)
                # Add Peptide
                mqrun.samples[sample_name].pept_dict[evidenceID] = pep

            # Add proteins to sample
            mqrun.samples[sample_name].prot_dict.update(tmp_prot)

        # Create sample Protein and Peptide lists
        for sample_name, sample_obj in mqrun.samples.items():
            sample_obj.update_prot_list()
            sample_obj.update_pept_list()

        infile.close()

        return mqrun, intensities
