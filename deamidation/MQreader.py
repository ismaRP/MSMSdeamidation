import os
import re
import numpy as np
import scipy as sp
from sklearn.preprocessing import quantile_transform
from deamidation.DSevidence import protein, sample, dataBatch
from deamidation.accFunctions import readHeader

class evidenceBatchReader():

    def __init__(self, datapath, ptm=['de'], aa=['QN'], sep='\t', byPos=False,
                 qt=False, sf_exp=[], include=None, exclude=None):
        """
        Include: only selected are analyzed
        Exclude: analyze all datasets but the ones set here
        If both include and exclude are set, only include will be used.
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
        self.ptm = ['\(' + p + '\)' for p in ptm]
        self.aa = aa
        self.datasets = kept_data
        self.paths = [datapath + d for d in self.datasets]
        self.sep = sep
        self.byPos = byPos
        self.qt = qt
        self.sf_exp = sf_exp

    def readBatch(self):

        tmp_batch = {}
        intensities = [[] for i in range(50)]
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

    def __getmod(self, seq, modseq, regexM, regexS, start=0):

        if self.byPos == False:
            d = [[s.group(1)+ '-' 'nan', m.group(2), 0]
                for s,m in zip(regexS.finditer(seq),  regexM.finditer(modseq))]
        else:
            # Find tripeps and positions
            d = [[s.group(1)+ '-' + str(s.start()+start), m.group(2), s.start()+start]
                for s,m in zip(regexS.finditer(seq),  regexM.finditer(modseq))]
        return(d)
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

    def __qt(self, folder, sample_field):
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
                transformed = quantile_transform(np.reshape(to_transform,(-1,1)),
                                                 axis=0, copy=True,
                                                 n_quantiles=len(to_transform),
                                                 output_distribution='normal')
                transformed = transformed.flatten()
                transformed = transformed - np.min(transformed)
                intensity_list[sample_start:i] = transformed
                intensity_list.append(intensity)

                sample_start = i
                old_sample = sample_name
        # Transform last sample
        to_transform = intensity_list[sample_start:i+1]
        transformed = quantile_transform(np.reshape(to_transform,(-1,1)),
                                         axis=0, copy=True,
                                         n_quantiles=len(to_transform),
                                         output_distribution='normal')
        transformed = transformed.flatten()
        transformed = transformed - np.min(transformed)
        intensity_list[sample_start:i+1] = transformed
        infile.close()
        return(intensity_list)

    def importEvidence(self, folder, intensities=[], sample_field='Raw file'):
        """
        Read evidence.txt file from MaxQuant output

        sampleProt =
            {sample1: sampleObject,
             ...
            }

        """
        # Read peptides.txt
        peptides = self.__read_peptides(folder)

        sampleProt = {}

        infile = open(folder + '/evidence.txt', 'r')
        header = infile.readline()[:-1].split(self.sep)

        headerPos = readHeader(['modseq', 'intensity', 'sample',
                                'protein', 'length',
                                'seq', 'peptideID', 'evidenceID'],
                               ['Modified sequence', 'Intensity', sample_field,
                                'Leading razor protein', 'Length',
                                'Sequence', 'Peptide ID', 'id'],
                               header)
        ### Complete regex
        aalist = '([ARNDCEQGHILKMFPUSTWYVX_]{1})'

        # Build regexM to find in modseq
        regexM = '|'.join(self.ptm)
        regexM = '(' + regexM + ')?'
        regexM = '([' + ''.join(self.aa) + '])' + regexM

        # Build regexS to find in seq
        regexS = '(?=(' + aalist + '([' + ''.join(self.aa) + '])' + aalist + '))'

        # Compile regex
        regexM = re.compile(regexM)
        regexS = re.compile(regexS)

        if self.qt == True:
            tr_intensities = self.__qt(folder, sample_field)

        for i, line in enumerate(infile):
            line = line[:-1]
            line = line.split(self.sep)

            sample_name = line[headerPos['sample']]
            protein_name = line[headerPos['protein']]
            modseq = line[headerPos['modseq']]
            # _GAP(hy)GADGPAGAP(hy)GTP(hy)GPQ(de)GIAGQ(de)R_
            seq = line[headerPos['seq']]
            length = line[headerPos['length']]
            length = int(length)
            evidenceID = line[headerPos['evidenceID']]

            if self.qt == False:
                intensity = line[headerPos['intensity']]
                intensity = float(intensity) if intensity!='' else 0
            else:
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

            deamid_tripep = self.__getmod('_'+seq+'_', modseq,
                                          regexM, regexS,
                                          start)

            # Fill sample and protein data
            if sample_name not in sampleProt:
                sampleProt[sample_name] = sample(sample_name)
                sampleProt[sample_name].proteins[protein_name] = protein(protein_name)
            elif protein_name not in sampleProt[sample_name].proteins:
                sampleProt[sample_name].proteins[protein_name] = protein(protein_name)

            sampleProt[sample_name].proteins[protein_name].seqs.update({seq})
            sampleProt[sample_name].proteins[protein_name].total_int += intensity
            sampleProt[sample_name].total_int += intensity
            if intensity > sampleProt[sample_name].max_int:
                sampleProt[sample_name].max_int = intensity
            if intensity < sampleProt[sample_name].min_int:
                sampleProt[sample_name].min_int = intensity

            # Collect tripetides X[NQ]Y into protein
            for dap in deamid_tripep:
                idx = 2 if dap[1] is None else 0
                trp = dap[0]
                position = dap[2]

                if trp[2] == '_':
                    trp = trp[:2] + aft + trp[3:]
                elif trp[0] == '_':
                    trp = bef + trp[1:]

                if trp not in sampleProt[sample_name].proteins[protein_name].tripeptides:
                    new_entry = np.array([0,0,0,0, position])
                    new_entry[idx] = intensity
                    new_entry[idx+1] = 1
                    sampleProt[sample_name].proteins[protein_name].tripeptides[trp] = new_entry
                else:
                    sampleProt[sample_name].proteins[protein_name].tripeptides[trp][idx] += intensity
                    sampleProt[sample_name].proteins[protein_name].tripeptides[trp][idx+1] += 1

        infile.close()

        return sampleProt, intensities
