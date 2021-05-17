import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def readHalftimes(N_file, Q_file):
    # Read in halftimes
    with open(N_file, 'r') as f:
        N_halftimes = json.load(f)

    with open(Q_file, 'r') as f:
        Q_halftimes = json.load(f)


    return(N_halftimes, Q_halftimes)


def plot_halftimes_hist(N_halftimes, Q_halftimes):
    nh = [r[0] for r in list(N_halftimes.values())]
    nh = np.array(nh)
    qh = [r[0] for r in list(Q_halftimes.values())]
    qh = np.array(qh)

    fig = plt.figure()
    axN = fig.add_subplot(211)
    axN.hist(np.log(nh), bins=30, density=True)
    axN.set_xlabel('N deamidation halftime')
    axQ = fig.add_subplot(212)
    axQ.hist(np.log(qh), bins=20, density=True)
    axQ.set_xlabel('Q deamidation halftime')

    plt.savefig('./deamidation_halftimes.png')
    plt.close()

def sort_tripeps(tripeps, halftimes):
    """
    Sort an array of tripeptides according to
    """
    tripeps = np.array(tripeps)
    hts = [halftimes[trp][0] for trp in tripeps]
    hts = np.array(hts)
    ind = np.argsort(hts)
    tripeps = tripeps[ind]
    hts = hts[ind]
    return tripeps, hts


def group_by_bins(halftimes, Nbins, Qbins):
    for tripep, data in halftimes.items():
        halftime = data[0]
        log_halftime = np.log(halftime)
        if tripep[1] == 'N':
            for i in range(len(Nbins)-1):
                if Nbins[i] <= log_halftime < Nbins[i+1]:
                    halftimes[tripep] = [halftime, i]
        if tripep[1] == 'Q':
            for i in range(len(Qbins)-1):
                if Qbins[i] <= log_halftime < Qbins[i+1]:
                    halftimes[tripep] = [halftime, i+len(Nbins)-1]

def group_by_seq(halftimes, pos='right'):
    for tripep, data in halftimes.items():
        if pos== 'right':
            group = '_' + tripep[1] + tripep[2]
        elif pos=='left':
            group = tripep[0] + tripep[1] + '_'
        elif pos=='middle':
            group = '_' + tripep[1] + '_'
        halftimes[tripep][1] = group

def group_by_range(halftimes, range=0.1):
    halftimes_tmp = halftimes.copy()
    # Mark tripeptides as unused for grouping
    for k in halftimes_tmp.keys():
        halftimes_tmp[k].append(0)
    group = 0
    for k in halftimes_tmp.keys():
        if halftimes_tmp[k][2]==1:
            continue # Continue if already grouped
        ht = halftimes_tmp[k][0]
        halftimes_tmp[k][1] = str(group)
        lower_limit = ht - (ht * range)
        upper_limit = ht + (ht * range)
        halftimes_tmp[k][2] = 1
        for l in halftimes_tmp.keys():
            if lower_limit<=halftimes_tmp[l][0]<upper_limit and halftimes_tmp[l][2]==0:
                halftimes_tmp[l][1] = str(group)
                halftimes_tmp[l][2] = 1 # Mark as used
        group += 1 # Next group
    for k in halftimes_tmp.keys():
        halftimes[k] = halftimes_tmp[k]


def box_plot_halftimes(halftimes):
    # Separate Q and N
    N_groups = {}
    Q_groups = {}

    for tripep, data in halftimes.items():
        if tripep[1] == 'N':
            gr = data[1]
            ht = data[0]
            if gr not in N_groups:
                N_groups[gr] =  [ht]
            else:
                N_groups[gr].append(ht)
        elif tripep[1] == 'Q':
            gr = data[1]
            ht = data[0]
            if gr not in Q_groups:
                Q_groups[gr] =  [ht]
            else:
                Q_groups[gr].append(ht)
    N_list = []
    Q_list = []
    for gr, hts in N_groups.items():
        N_list.append(hts)
    for gr, hts in Q_groups.items():
        Q_list.append(hts)

    fig = plt.figure()
    axN = fig.add_subplot(211)
    axQ = fig.add_subplot(212)
    axN.boxplot(N_list)
    axQ.boxplot(Q_list)
    axN.semilogy()
    axQ.semilogy()
    plt.savefig('./halftimes_boxplot.png')
    plt.close()



def readHeader(names_after, names_before, header):
    """
    names_after and names_before are lists that must match
    """

    headerPos = {}
    for i in range(len(names_after)):
        headerPos[names_after[i]] = header.index(names_before[i])
    return(headerPos)


def readSampleInfo(path):
    sampleInfo = {}
    with open(path, 'r') as infile:
        header = infile.readline()[:-1].split('\t')
        # headerPos = readHeader(['sample', 'age', 'sp',
        #                         'genus', 'loc', 'site', 'PartBody'],
        #                        ['SampleName', 'Age', 'Species label',
        #                         'Genus', 'Location', 'Site', 'Part of Body'],
        #                        header)

        for line in infile:
            line = line[:-1]
            line = line.split('#')[0].rstrip()
            if line == '':
                continue
            line = line.split('\t')
            sample = line[0]
            fields = line[1:]
            sampleInfo[sample] = fields
    return sampleInfo, header

def readProtList(path):
    with open(path, 'r') as infile:
        proteins = {}
        for line in infile.readlines():
            line = line.rstrip()
            line = line.split('#')[0].rstrip()
            if line == '':
                continue
            line = line.split('\t')
            ch1_start = int(line[2]) if line[2]!='NA' else 1
            ch2_start = int(line[3]) if line[3]!='NA' else np.inf
            proteins[line[0]]=[line[1], ch1_start, ch2_start]
        return(proteins)

def map_range(sum_int, sample_max_int, sample_min_int):
    """
    Maps intensity to range given the maximum of the sample
    """

    n_max = 500
    n_min = n_max/10
    n_range= n_max-n_min
    int_range = sample_max_int - sample_min_int
    if int_range == 0:
        norm_int = n_min
    else:
        norm_int = (((sum_int - sample_min_int)*n_range)/int_range) + n_min

    return norm_int


def pca(data, k, inv=False):
    """
    input: datamatrix. Rows must be observations while columns variables
    output:  1) the eigenvalues in a vector (numpy array) in descending order
             2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector
                (in the same order as its associated eigenvalue)
    note: make sure the order of the eigenvalues (the projected variance)
    is decreasing, and the eigenvectors have the same order as their associated eigenvalues
    """

    data = data - np.mean(data, axis=0)
    # Calculate covariance matrix
    cov_mat = np.cov(data, rowvar=False)
    # Calculate eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(cov_mat)
    # Get index of sorted eigenvalues
    e_index = (-evals).argsort()
    # Sort eigenvalues
    sorted_evals = np.sort(-evals)
    sorted_evals = -sorted_evals
    # Sort eigenvectors
    sorted_evecs = evecs[:,e_index]
    if inv:
        sorted_evecs = sorted_evecs * (-1)
    # Take first k eigenvectors
    evecs_red = sorted_evecs[:,0:k]
    # Change basis, using eigenvectors
    projdata = np.dot(data, evecs_red)

    return projdata, sorted_evecs, sorted_evals
