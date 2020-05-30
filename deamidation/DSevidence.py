import os
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from deamidation.accFunctions import map_range
import warnings

class protein():

    def __init__(self, name):
        """
        tripeptides =
        {AQG: [deamid_int, deamid_count,
               nondeamid_int, nondeamid_count,
               pos],
         ...
        }
        """
        self.name = name
        self.tripeptides = {}
        self.seqs = set()
        self.total_int = 0


class sample():

    def __init__(self, name):
        self.name = name
        self.total_int = 0
        self.max_int = 0
        self.min_int = np.inf
        self.proteins = {}

class dataBatch():

    def __init__(self, tmp_batch, intensities, byPos):

        self.byPos = byPos
        self.intensities = intensities

        samples = []

        for ev in tmp_batch.values():
            for s in ev.values():
                samples.append(s)

        self.samples = samples


    def get_sampleTripeps(self, sampleInfo, proteinInfo,
                          norm_type='simple', filter=None):
        """
        Collapses tripeptide deamidation per sample, making a weighterd average
        across the selected proteins present in proteinInfo
        {sample: {COL1A1-ANG-400: [avg.rel_deamid, counts,
                                   norm_int,
                                   prot_id, chain,
                                   pos,
                                   ],
                  ...,
                  },
         ...}

        """
        sampleTripeps = {}

        for sample in self.samples:
            sample_name = sample.name
            if sample_name not in sampleInfo:
                continue
            sampleTripeps[sample_name] = {}
            # Loop over the proteins to filter out the ones not in
            # proteinInfo and get total intenity
            for prot in sample.proteins.values():
                prot_name = prot.name
                if prot_name not in proteinInfo:
                    continue
                for tripep, tripep_data in prot.tripeptides.items():
                    if filter != None:
                        if tripep[1] != filter:
                            continue
                    chain = proteinInfo[prot_name][0]
                    raw_pos = tripep_data[4]
                    sum_int = tripep_data[0] + tripep_data[2]
                    sum_counts = tripep_data[1] + tripep_data[3]
                    tripep = chain + '-' + tripep # Attach chain
                    if sum_int != 0:
                        rel_int = tripep_data[0]/sum_int
                        if norm_type == 'range':
                            norm_fact = map_range(
                                sum_int, sample.max_int, sample.min_int
                            )
                        else:
                            norm_fact = sum_int/sample.total_int
                        if tripep not in sampleTripeps[sample_name]:
                            sampleTripeps[sample_name][tripep]=[
                                0, 0, 0, prot_name, chain, raw_pos
                            ]
                        sampleTripeps[sample_name][tripep][0] += rel_int
                        sampleTripeps[sample_name][tripep][1] += sum_counts
                        sampleTripeps[sample_name][tripep][2] += norm_fact
        return sampleTripeps


    def plot_intensities(self, path=None):

        #plot length vs intensity
        empty = np.array([True if len(v)==0 else False for v in self.intensities])
        intensities = self.intensities[~empty]
        lengths = np.array(range(1,51))[~empty]
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
        ax.semilogy()
        plt.legend()
        plt.title('Intensity distribution by length')

        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()



class deamidationMatrix():
    """
    Atributes:
        - D
        - Ydata
        - rel_ints
        - counts
        - trps_data
            (prot_name, seq,  pos, corr_pos 'prot_name-seq-pos')
        - layers. List of matrices aligned to D
    """
    def __get_trps_data(self, sampleTripeps):
        all_tripeps = {}
        for sample, tripeps in sampleTripeps.items():
            # print('{}: {}'.format(sample,tripeps.keys()))
            for trp, values in tripeps.items():
                if trp not in all_tripeps:
                    # [avg.rel_deamid, counts, norm_int, prot_name, pos, corrected_pos]
                    seq = trp.split('-')[1]
                    all_tripeps[trp] = (values[3], values[4], seq, values[5], 0)
        trps_data = []
        for trip, values in all_tripeps.items():
            trps_data.append(values + tuple([trip]))
        trps_data = np.array(trps_data,
                             dtype=[
                                ('prot_id', '<U60'),
                                ('prot_name', '<U10'),
                                ('tripep', 'U3'),
                                ('position', 'i4'),
                                ('corr_pos', 'i4'),
                                ('trp_chain', '<U20')])
        trps_data = np.sort(trps_data, 0, order=['tripep', 'prot_name', 'position'])
        return trps_data

    def __set_matrix(self, sampleTripeps, sampleInfo):
        D = []
        rel_ints = []
        Ydata = {}
        counts = []
        for h in self.Ykeys:
            Ydata[h] = np.array([])
        for sample, tripeps in sampleTripeps.items():
            d = []
            counts_line = []
            relint_line = []
            for trp in self.trps_data:
                if trp[5] in tripeps:
                    d.append(tripeps[trp[5]][0])
                    counts_line.append(tripeps[trp[5]][1])
                    relint_line.append(tripeps[trp[5]][2])
                else:
                    d.append(np.nan)
                    counts_line.append(0)
                    relint_line.append(0)
            D.append(d)
            counts.append(counts_line)
            rel_ints.append(relint_line)
            for i, h in enumerate(self.Ykeys[1:]):
                Ydata[h] = np.append(Ydata[h], sampleInfo[sample][i])
            Ydata[self.Ykeys[0]] = np.append(Ydata[self.Ykeys[0]], sample)
        D = np.array(D)
        rel_ints = np.array(rel_ints)
        counts = np.array(counts)
        self.D = D
        self.Ydata = Ydata
        self.counts = counts
        self.rel_ints = rel_ints

    def __init__(self, sampleTripeps=None, sampleInfo=None, header=None,
                 D=None, Ydata=None, rel_ints=None, trps_data=None, counts=None,
                 layers=[]):
        if ((sampleInfo is not None) and (header is not None)
            and (sampleTripeps is not None)):
            #
            self.Ykeys = header
            self.trps_data = self.__get_trps_data(sampleTripeps)
            self.__set_matrix(sampleTripeps, sampleInfo)
            self.simulated_data = False
            self.layers = []
        elif ((D is not None) and (Ydata is not None)
            and (trps_data is not None)):
            #
            self.D = D
            self.Ydata = Ydata
            self.counts = counts
            self.rel_ints = rel_ints
            self.trps_data = trps_data
            self.layers = layers
            if (counts is None) and (rel_ints is None):
                self.simulated_data = True
            else:
                self.simulated_data = False
        else:
            raise ValueError('Provide either a sampleTripep data structure or\
                              D, Ydata, counts, rel_ints and trps_data')

    def write_csv(self, path, dfile='deamidation_matrix.csv', cfile='counts_matrix.csv'):
        """
        Write .csv files of D and counts
        """
        Ddata_toprint = np.array(self.D, dtype = '<U42')
        Ddata_toprint = np.hstack((self.Ydata['SampleName'].reshape(-1,1),
                                   Ddata_toprint))
        if not self.simulated_data:
            counts_toprint = np.array(self.counts, dtype = '<U42')
            counts_toprint = np.hstack((self.Ydata['SampleName'].reshape(-1,1),
                                       counts_toprint))

        Ddata_outf = open(path+'/'+dfile, 'w+')
        Ddata_outf.write(','.join(np.append(['tripep'], self.trps_data['tripep']))+'\n')
        Ddata_outf.write(','.join(np.append(['position'], self.trps_data['position']))+'\n')
        Ddata_outf.write(','.join(np.append(['corr pos'], self.trps_data['corr_pos']))+'\n')
        Ddata_outf.write(','.join(np.append(['prot name'], self.trps_data['prot_name']))+'\n')

        if not self.simulated_data:
            counts_outf = open(path+'/'+cfile, 'w+')
            counts_outf.write(','.join(np.append(['tripep'], self.trps_data['tripep']))+'\n')
            counts_outf.write(','.join(np.append(['position'], self.trps_data['position']))+'\n')
            counts_outf.write(','.join(np.append(['corr pos'], self.trps_data['corr_pos']))+'\n')
            counts_outf.write(','.join(np.append(['prot name'], self.trps_data['prot_name']))+'\n')

        if not self.simulated_data:
            for dline, cline in zip(Ddata_toprint, counts_toprint):
                dline = ','.join(dline)
                cline = ','.join(cline)
                Ddata_outf.write(dline+'\n')
                counts_outf.write(cline+'\n')
            Ddata_outf.close()
            counts_outf.close()
        else:
            for dline in Ddata_toprint:
                dline = ','.join(dline)
                Ddata_outf.write(dline+'\n')
            Ddata_outf.close()

    def set_trps_data(self, trps_data):
        self.trps_data = trps_data


    def log_1_d(self, tol):
        """
        Transform data to ln(1-d) and store it in a layer
        """
        # Convert to nan d > 1-tol
        L = np.copy(self.D)
        L[L > 1-tol] = np.nan
        L = np.log(1-L)
        self.layers.append(L)

    def filter_by_pwcounts(self, cutoff=5):
        """
        Returns a mask matrix indicating which tripeps to keep
        after going pairwise through them and removing samples with
        low counts.
        """
        if self.simulated_data:
            raise ValueError('Simulated data does not contain counts info.')
        num_tripeps = len(self.trps_data)
        info_matrix = np.zeros((num_tripeps,num_tripeps))
        for i in range(num_tripeps):
            for j in range(num_tripeps):
                counts_i = self.counts[:,i]
                counts_j = self.counts[:,j]
                pairs = np.logical_and(counts_i >= cutoff, counts_j >= cutoff)
                if np.sum(pairs) > 10:
                    info_matrix[i,j] = True
                else:
                    info_matrix[i,j] = False
        return np.sum(info_matrix, axis=0) > 1


    def sort_tripeps(self, idx):
        sorted_D = self.D[:,idx]
        sorted_trps_data = self.trps_data[idx]
        sorted_layers = [None]*len(self.layers)
        for i in range(len(self.layers)):
            sorted_layers[i] = self.layers[i][:,idx]
        if not self.simulated_data:
            sorted_counts = self.counts[:,idx]
            sorted_rel_ints = self.rel_ints[:,idx]
        else:
            sorted_counts = None
            sorted_rel_ints = None
        deamid_mat = deamidationMatrix(D=sorted_D,
                                       Ydata=self.Ydata,
                                       counts=sorted_counts,
                                       rel_ints=sorted_rel_ints,
                                       trps_data=sorted_trps_data,
                                       layers=sorted_layers)
        return(deamid_mat)


    def filter_tripeps(self, mask):
        filt_D = self.D[:,mask]
        filt_trps_data = self.trps_data[mask]
        filt_layers = [None]*len(self.layers)
        for i in range(len(self.layers)):
            filt_layers[i] = self.layers[i][:,mask]
        if not self.simulated_data:
            filt_counts = self.counts[:,mask]
            filt_rel_ints = self.rel_ints[:,mask]
        else:
            filt_counts = None
            filt_rel_ints = None
        deamid_mat = deamidationMatrix(D=filt_D,
                                       Ydata=self.Ydata,
                                       counts=filt_counts,
                                       rel_ints=filt_rel_ints,
                                       trps_data=filt_trps_data,
                                       layers=filt_layers)
        return(deamid_mat)

    def filter_samples(self, mask):
        filt_D = self.D[mask,:]
        filt_Ydata = {}
        filt_layers = [None]*len(self.layers)
        for i in range(len(self.layers)):
            filt_layers[i] = self.layers[i][mask,:]
        if not self.simulated_data:
            filt_counts = self.counts[mask,:]
            filt_rel_ints = self.rel_ints[mask,:]
        else:
            filt_counts = None
            filt_rel_ints = None
        for k, v in self.Ydata.items():
            filt_Ydata[k] = v[mask]
        deamid_mat = deamidationMatrix(D=filt_D,
                                       Ydata=filt_Ydata,
                                       counts=filt_counts,
                                       rel_ints=filt_rel_ints,
                                       trps_data=self.trps_data,
                                       layers=filt_layers)
        return(deamid_mat)

    def merge_by_tripep(self):

        if len(self.layers) > 0:
            warnings.warn('The layers of transformed data will be removed.')

        groups = self.trps_data['tripep']

        groups_set = np.sort(list(set(groups)))
        new_D = []
        new_trps_data = []
        new_counts = []
        new_rel_ints = []

        comb_mat = np.nan_to_num(self.D) * self.rel_ints
        for gr in groups_set:
            mask = groups == gr
            num = np.sum(comb_mat[:,mask], axis=1)
            if not self.simulated_data:
                sum_rel_ints = np.sum(self.rel_ints[:,mask], axis=1)
                sum_counts = np.sum(self.counts[:,mask], axis=1)
                new_rel_ints.append(sum_rel_ints)
                new_counts.append(sum_counts)
            new_D.append(num/sum_rel_ints)

            new_trps_data.append(('NA', 'NA', gr, 0, 0, 'NA-'+ gr + '-0'))
        new_D = np.array(new_D).T
        new_trps_data = np.array(new_trps_data,
                                 dtype=[
                                    ('prot_id', '<U35'),
                                    ('prot_name', '<U10'),
                                    ('tripep', 'U3'),
                                    ('position', 'i4'),
                                    ('corr_pos', 'i4'),
                                    ('string', '<U45')
                                    ])
        if not self.simulated_data:
            new_counts = np.array(new_counts).T
            new_rel_ints = np.array(new_rel_ints).T
        else:
            new_counts = None
            new_rel_ints = None

        new_deamid_mat = deamidationMatrix(D=new_D,
                                           Ydata=self.Ydata,
                                           counts=new_counts,
                                           rel_ints=new_rel_ints,
                                           trps_data=new_trps_data)
        return new_deamid_mat


    def merge_by_pos(self, corr):

        if len(self.layers) > 0:
            warnings.warn('The layers of transformed data will be removed.')

        if corr == True:
            groups = self.trps_data['corr_pos']
        else:
            groups = self.trps_data['position']
        groups_set = np.sort(list(set(groups)))

        prot_names = self.trps_data['prot_name']
        prot_set = np.sort(list(set(prot_names)))

        new_D = []
        new_trps_data = []
        new_counts = []
        new_rel_ints = []

        comb_mat = np.nan_to_num(self.D) * self.rel_ints
        for pr in prot_set:
            for gr in groups_set:
                mask = np.logical_and(prot_names == pr, groups == gr)
                if np.sum(mask) == 0:
                    continue
                num = np.sum(comb_mat[:,mask], axis=1)
                if not self.simulated_data:
                    sum_rel_ints = np.sum(self.rel_ints[:,mask], axis=1)
                    sum_counts = np.sum(self.counts[:,mask], axis=1)
                    new_rel_ints.append(sum_rel_ints)
                    new_counts.append(sum_counts)
                new_D.append(num/sum_rel_ints)

                trp = self.trps_data[mask][0]['tripep']
                new_trps_data.append((pr, pr, trp, gr, gr,
                                      pr + '-' + trp + '-' + str(gr)))
        new_D = np.array(new_D).T
        new_trps_data = np.array(new_trps_data,
                                 dtype=[
                                    ('prot_id', '<U35'),
                                    ('prot_name', '<U10'),
                                    ('tripep', 'U3'),
                                    ('position', 'i4'),
                                    ('corr_pos', 'i4'),
                                    ('string', '<U45')
                                    ])
        if not self.simulated_data:
            new_counts = np.array(new_counts).T
            new_rel_ints = np.array(new_rel_ints).T
        else:
            new_counts = None
            new_rel_ints = None

        new_deamid_mat = deamidationMatrix(D=new_D,
                                           Ydata=self.Ydata,
                                           counts=new_counts,
                                           rel_ints=new_rel_ints,
                                           trps_data=new_trps_data)
        return new_deamid_mat

    def influx(self):
        """
        Calculate information influx for each tripeptide
        """

        num_dims = self.D.shape[1]
        Ddata = self.D
        Dresp = np.zeros(self.D.shape)
        Dresp[~np.isnan(Ddata)] = 1
        influx = []
        prop_missing = []
        for j in range(num_dims):
            num = 0
            den = 0
            dresp_j = Dresp[:,j]
            p = np.sum(1-dresp_j)/len(dresp_j)
            prop_missing.append(p)
            for k in range(num_dims):
                dresp_k = Dresp[:,k]
                If_jk = (1-dresp_j) * dresp_k # element-wise product
                num += np.sum(If_jk)
                den += np.sum(dresp_k)
            influx.append(num/den)
        return np.array(influx), np.array(prop_missing)



    def correct_pos(self, prot_info):
        """
        Correct positions
        """
        trps_data = self.trps_data
        new_trps_data = []
        for trp in trps_data:
            chain = prot_info[trp['prot_id']][0]
            ch1_start = prot_info[trp['prot_id']][1]
            ch1_end = prot_info[trp['prot_id']][2]
            if chain =='COL1A' and trp['position'] < ch1_end:
                chain = chain + '1'
            trp['prot_name'] = chain
            corr_pos = trp['position'] - ch1_start
            corr_end = ch1_end - ch1_start
            if corr_pos<0:
                corr_pos = 0
            if corr_pos > corr_end:
                corr_pos = 0
            trp['corr_pos'] = corr_pos
            new_trps_data.append(trp)
        new_trps_data = np.array(new_trps_data,
                             dtype=[
                                ('prot_id', '<U35'),
                                ('prot_name', '<U10'),
                                ('tripep', 'U3'),
                                ('position', 'i4'),
                                ('corr_pos', 'i4'),
                                ('string', '<U45')
                                ])
        self.set_trps_data(new_trps_data)

    def stagger_pos(self, l_period):
        """
        Calculate stagger. Positions are assumet to be corrected
        """
        trps_data = self.trps_data

        n_period = np.floor(trps_data['corr_pos']/l_period)
        trps_data['corr_pos'] = trps_data['corr_pos'] - l_period * n_period

        self.set_trps_data(trps_data)

        return n_period.astype('int') + 1


    def correct_stagger_pos(self, prot_info, stagger):
        trps_data = self.trps_data
        new_trps_data = []
        for trp in trps_data:
            chain = prot_info[trp['prot_id']][0]
            ch1_start = prot_info[trp['prot_id']][1]
            trp['prot_name'] = chain
            corr_pos = trp['position'] - ch1_start
            if corr_pos<0:
                corr_pos = 0
            n_period = np.floor(corr_pos/stagger)
            corr_pos = corr_pos - stagger * n_period
            trp['corr_pos'] = corr_pos
            new_trps_data.append(trp)
        new_trps_data = np.array(new_trps_data,
                             dtype=[
                                ('prot_id', '<U35'),
                                ('prot_name', '<U10'),
                                ('tripep', 'U3'),
                                ('position', 'i4'),
                                ('corr_pos', 'i4'),
                                ('string', '<U45')
                                ])
        self.set_trps_data(new_trps_data)
