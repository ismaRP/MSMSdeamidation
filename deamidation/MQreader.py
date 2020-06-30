import os
import re
import numpy as np
import scipy as sp
from sklearn.preprocessing import quantile_transform
from deamidation.DSevidence import peptide, protein, sample, dataBatch
from deamidation.accFunctions import readHeader
from deamidation.accFunctions import readSampleInfo, readProtList

class evidenceBatchReader():

    def __init__(self, datapath, prot_f, samples_f,
                 ptm=['de'], aa=['QN'], sep='\t',
                 tr=None, sf_exp=[], include=None, exclude=None):
        """
        Include: only selected are analyzed
        Exclude: analyze all datasets but the ones set here
        If both include and exclude are set, only include will be used.

        tr: ['qt', 'lognorm', 'log']
        """
        if datapath[-1] != '/':
            datapath += '/'
        datasets = os.listdir(datapath)
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
        aamod = [a + mod for a,mod in zip(aa_m, ptm)]
        regexM = '(' + '|'.join(aamod) + ')'

        # Compile regex
        self.regexM = re.compile(regexM)
        self.regexS = re.compile(regexS)

        self.datasets = kept_data
        self.paths = [datapath + d for d in self.datasets]
        self.sep = sep
        self.sf_exp = sf_exp

        self.prot_f = readProtList(prot_f)
        self.samples_f = readSampleInfo(samples_f)
            doc = prot_f property."
            def fget(self):
                return sprot_f
            def fset(self, value):
                sprot_f = value
            def fdel(self):
                del sprot_f
            return localsprot_f = properprot_f())
        if tr in {'qt','lognorm'}:
            self.per_sample_tr = True
            if tr == 'qt':
                self.tr = tr
                self.tr_f = self.__qt
            elif tr == 'lognor':
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

        tmp_batch = {}
        intensities = [[] for i in range(60)]
        for d, p in zip(self.datasets, self.paths):
            if d in self.sf_exp:
                tmp_batch[d], intensities = self.importEvidence(p, intensities,
                                                              sample_field='Experiment')
            else:
                tmp_batch[d], intensities = self.importEvidence(p, intensities,
                                                              sample_field='Raw file')
        intensities = np.array([np.array(v) for v in intensities])

        data_batch = dataBatch(tmp_batch, intensities, self.byPos)

        return(data_batch)


    def __read_peptides(self, folder, sep='\t'):
        """
        Reads peptides.txt file from MaxQuant output into a
        simple dictionary structure:
        {peptide id: [start_position, end_position, bef, aft]}
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

    def __getmod(self, seq, modseq, start=0):
        seq = '_' + seq + '_'
        # PTMs ids in peptides are defined by residue + local position
        # [
        #    [
        #         residue,
        #         modification,
        #         local position,
        #         protein position
        #     ],
        # ...
        # ]
        ptms = [[s.group(3), m.group(0), s.start(3), s.start(3)+start]
                 for s,m in zip(regexS.finditer(seq),regexM.finditer(modseq))]
        return(ptms)
    # def __getmod(self, seq, modseq, deamidRe, tripepRe, start=0):
    #     """
    #     Parse sequence and modified sequence.
    #     Returns a list of lists:
    #     [[tripep1, '(de)', position], [tripep2, '(de)', position]]
    #     if byPos is True, tripep has the format like in "AQG-532"
    #     if byPos is False, tripep is just AQG and position is 0
    #     """
    #
    #     if self.byPos == False:
    #         d = [[m.group(2)+m.group(4)+m.group(6)+'-nan',
    #               m.group(5),
    #               0]
    #              for m in deamidRe.finditer(modseq)]
    #     else:
    #         # Find tripeps in the non-modified sequence
    #         pos = [m.start()+start for m in tripepRe.finditer(seq)]
    #         # Find tripeps in the modified sequence
    #         d = [
    #                 [m.group(2)+m.group(4)+m.group(6)+'-'+str(pos[i]),
    #                  m.group(5),
    #                  pos[i]
    #             ]for i, m in enumerate(deamidRe.finditer(modseq))]
    #     return d

    def __qt(self, x):
        transformed = quantile_transform(np.reshape(x,(-1,1)),
                                         axis=0, copy=True,
                                         n_quantiles=len(x),
                                         output_distribution='normal')
        transformed = transformed.flatten()
        transformed = transformed - np.min(transformed)
        return(transformed)

    def __log_norm(self, x):
        return(np.log(x/np.sum(x)+1))

    def __log(self, x):
        return(np.log(x+1))

    def __tr_per_sample(self, folder, sample_field, **kargs):
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
            intensity = float(intensity) if intensity!='' else 0
            if sample_name == old_sample or old_sample == None:
                intensity_list.append(intensity)
                old_sample = sample_name
                continue
            else:

                to_transform = intensity_list[sample_start:i]
                transformed = transform = self.tr_f(to_transform)

                intensity_list[sample_start:i] = transformed
                intensity_list.append(intensity)

                sample_start = i
                old_sample = sample_name
        # Transform last sample
        to_transform = intensity_list[sample_start:i+1]
        transformed = transform = self.tr_f(to_transform)

        intensity_list[sample_start:i+1] = transformed
        infile.close()
        return(intensity_list)


    def __tr_one_shot(self, folder, sample_field, **kargs):
        infile = open(folder + '/evidence.txt', 'r')
        header = infile.readline()[:-1].split(self.sep)

        headerPos = readHeader(['intensity'],
                               ['Intensity'],
                               header)
        old_sample = None
        sample_start = 0
        intensity_list = []
        # Collect intensities
        for line in infile:
            line = line[:-1]
            line = line.split(self.sep)
            intensity = line[headerPos['intensity']]
            intensity = float(intensity) if intensity!='' else 0
            intensity_list.append(intensity)
        intensity_list = np.array(intensity_list)
        if self.tr is not None:
            intensity_list = self.tr_f(intensity_list)
        return(intensity_list)



    def importEvidence(self, folder, intensities=[], sample_field='Raw file'):
        """
        Read evidence.txt file from MaxQuant output

        MQdata =
            {sample1: sampleObject,
             ...
            }

        """
        # Read peptides.txt
        peptides = self.__read_peptides(folder)

        MQdata = {} # Contains samples with their peptides
        # {
        #     prot1id: prot1_object,
        #     prot2id: prot2_object
        # }

        infile = open(folder + '/evidence.txt', 'r')
        header = infile.readline()[:-1].split(self.sep)

        headerPos = readHeader(
            [
                'seq'
                'modseq',
                'intensity',
                'sample',
                'proteins',
                'leading_prots',
                'leading_razor_prot'
                'peptideID',
                'evidenceID'
            ],
            [
                'Sequence'
                'Modified sequence',
                'Intensity',
                sample_field,
                'Proteins',
                'Leading proteins',
                'Leading razor protein',
                'Peptide ID',
                'id'
            ],
            header)


        if self.tr is not None:
            if self.per_sample_tr:
                tr_intensities = self.__tr_per_sample(folder, sample_field)
            elif not self.per_sample_tr:
                tr_intensities = self.__tr_one_shot(folder, sample_field)
        else:
            tr_intensities = self.__tr_one_shot(folder, sample_field)


        for i, line in enumerate(infile):
            line = line[:-1]
            line = line.split(self.sep)

            sample_name = line[headerPos['sample']]

            modseq = line[headerPos['modseq']]
            # _GAP(hy)GADGPAGAP(hy)GTP(hy)GPQ(de)GIAGQ(de)R_
            seq = line[headerPos['seq']]
            length = line[headerPos['length']]
            length = int(length)
            evidenceID = line[headerPos['evidenceID']]

            # Get intensity from transofrmed list
            intensity = tr_intensities[i]

            peptideID = line[headerPos['peptideID']]

            # Get peptide-in-protein position info
            start = peptides[peptideID][0]
            start = int(start) if start!='' else 0

            bef = peptides[peptideID][2]
            if bef=='-' or bef=='': bef = '_'
            aft = peptides[peptideID][3]
            if aft=='-' or bef=='': aft = '_'

            intensities[length-1].append(intensity)

            proteins = set(line[headerPos['proteins']].split(';'))
            leading_prots = set(line[headerPos['leading_prots']].split(';'))
            leading_razor_prot = line[headerPos['leading_razor_prot']]



            ptms = self.__getmod(seq, modseq, start)
            pep = peptide(peptideID, intensity, start, end, sequence, ptms,
                          proteins, leading_prots, leading_razor_prot)

            # Fill sample and protein data
            if sample_name not in MQdata:
            # If new sample, create and add first peptide
                MQdata[sample_name] = sample(sample_name)
                MQdata[sample_name].peptides[seq] = p
            else:
            # Else (sample already created)
                # Add peptide
                MQdata[sample_name].peptides[seq] = p
            # Add proteins to sample
            for prot_id in proteins:
                if prot not in MQdata[sample_name].proteins:
                    prot_name = self.prot_f[prot_id][0]
                    MQdata[sample_name].proteins[prot] = protein(
                        prot_id, prot_name, len(MQdata[sample_name].proteins[prot]))
                    )

            MQdata[sample_name].total_int += intensity
            if intensity > MQdata[sample_name].max_int:
                MQdata[sample_name].max_int = intensity
            if intensity < MQdata[sample_name].min_int:
                MQdata[sample_name].min_int = intensity

        infile.close()


        return MQdata, intensities
