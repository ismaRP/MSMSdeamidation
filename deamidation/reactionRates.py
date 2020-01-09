import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from scipy import stats
from scipy.stats import truncnorm
import matplotlib.ticker as ticker


import networkx as nx


# General options
plt.style.use('seaborn-dark-palette')


def simulate_deamidation(K, sigmas, t, lim):
    # Draw times from a power distribution
    M = len(K)
    N = len(t)
    T = np.diag(t)
    K = np.expand_dims(K, 0) # Add another dimension to K
    K = np.repeat(K, N, 0)
    means = 1-np.exp(-np.dot(T,K))
    # Sample from truncated normal distr to avoid >1 and <0
    a = (0-means)/sigmas; b = (1-lim-means)/sigmas
    D = truncnorm.rvs(a, b, means, sigmas, (N,M))
    return D

def data_vs_sim_hist(data, sim, trps, xlabel, xrange=None, path=None, base_name=None):

    # Check if it's a N x M deamidation matrix
    # or a M length rank or R values
    if np.ndim(data) == 1:
        M = np.shape(data)[0]
        deamid_data = False
    elif np.ndim(data) == 2 and np.shape(data)[0] > 1:
        M = np.shape(data)[1]
        deamid_data = True
    elif np.ndim(data) == 2 and np.shape(data)[0] == 1:
        # Case of aggregated Q or N data treated as if
        # we had R or rank data
        M = 1
        deamid_data = False

    if M % 3 == 0:
        ncol = 3
        nrow = int(M/3)
    elif M % 4 == 0:
        ncol = 4
        nrow = int(M/4)
    elif M == 1:
        ncol = 1
        nrow = 1
    else:
        ncol = 3
        nrow = int(M/3) + 1

    fig = plt.figure(figsize=(9.6, 7.2), dpi=300)
    axes = fig.subplots(nrow, ncol, sharey='row', sharex=True, squeeze=False)
    for i in range(M):
        if deamid_data == True:
            d = data[:,i]; s = sim[:,i]
            d = d[~np.isnan(d)]
            s = s[~np.isnan(s)]
        elif deamid_data == False:
            d = data[i]; s = sim[i]
            d = d[~np.isnan(d)]
            s = s[~np.isnan(s)]

        axes_inds = np.unravel_index(i, axes.shape)
        axes[axes_inds].hist(
            [d, s], bins=15,
            range=xrange, density=True,
            color=['b', 'g'], alpha=0.7,
            label=['Data', 'Sim']
        )
        axes[axes_inds].set_title(trps[i], pad=-13)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor=(1,1,1), top=False,
                    bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.1))
    fig.savefig(path + base_name + '_data_vs_sim_hist.png')
    plt.close()




def R_QQplot(R_x, R_y, axes_info, tripeps, path=None, base_name=None,
             aspect='equal'):
    """
    Metadata is a list containing which axis contains simulated and which
    real data
    """
    # Find out if data is aggregated or not

    if np.ndim(R_x) == 2 and np.shape(R_x)[0] == 1:
        aggr=True
        M = 1
    elif np.shape(R_x)[0] > 1:
        M = len(R_x)
        aggr = False

    for i in range(M):
        x_row = np.delete(R_x[i], i)
        y_row = np.delete(R_y[i], i)
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        x_row = np.sort(x_row) # Long
        y_row = np.sort(y_row) # Short
        qlevels_x = np.arange(len(x_row),dtype=float)/len(x_row)
        qlevels_y = np.arange(len(y_row),dtype=float)/len(y_row)
        # The smallest array will set the quantiles. Interpolate in the larger
        if len(y_row) < len(x_row):
            qlevels = qlevels_y
            q_y = y_row
            q_x = np.interp(qlevels, qlevels_x, x_row)
        elif len(x_row) < len(y_row):
            qlevels = qlevels_x
            q_x = x_row
            q_y = np.interp(qlevels, qlevels_y, y_row)
        ax.scatter(q_x, q_y)
        if aspect == 'equal':
            maxval = max(x_row[-1], y_row[-1])
            minval = min(x_row[0], y_row[0])
            ax.plot([minval,maxval],[minval,maxval],'k-')
            ax.set_aspect('equal')
        elif aspect == 'num':
            aspt = q_x[-1]/q_y[-1]
            ax.set_aspect(aspt)
        else:
            maxval = max(x_row[-1], y_row[-1])
            minval = min(x_row[0], y_row[0])
            ax.plot([minval,maxval],[minval,maxval],'k-')
        plt.subplots_adjust(
            left = 0.135,  # the left side of the subplots of the figure
            right = 0.93 ,  # the right side of the subplots of the figure
            bottom = 0.1 , # the bottom of the subplots of the figure
            top = 0.9  ,   # the top of the subplots of the figure
        )
        # fig.tight_layout()
        ax.tick_params(labelsize='large', length=4)
        if x_row[-1] > 99:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        if y_row[-1] > 99:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        if aggr == True:
            ax.set_xlabel('Quantiles. ' + axes_info[0] + ' ' + tripeps[0],
                           size='x-large')
            ax.set_ylabel('Quantiles. ' + axes_info[1] + ' ' + tripeps[1],
                           size='x-large')
            if path is not None and base_name is not None:
                plt.savefig('{}QQ_{}.png'.format(path, base_name))
                plt.close()
            else:
                plt.show()
        else:
            ax.set_xlabel('Quantiles. ' + axes_info[0] + ' ' + tripeps[i],
                           size='x-large')
            ax.set_ylabel('Quantiles. ' + axes_info[1] + ' ' + tripeps[i],
                           size='x-large')
            if path is not None:
                plt.savefig('{}QQ_{}.png'.format(path, tripeps[i]))
                plt.close()
            else:
                plt.show()


def delta_lambda(A, B):
    mask = np.logical_and(A!=1, B!=1)
    mask_nonzeroA = A!=0
    mask = np.logical_and(mask, mask_nonzeroA)
    Anonone = A[mask]
    Bnonone = B[mask]
    logAnonone = np.log(1-Anonone)
    logBnonone = np.log(1-Bnonone)
    DL = (logAnonone - logBnonone) / logAnonone

    return np.sort(DL)

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value



def calc_Ri(D, counts=None, low_counts=None, log_tr=False, err_tol=1e-5, num_tol=0.09):
    M = D.shape[1]
    R = []

    if counts is not None and low_counts is not None:
        counts_mask = counts < low_counts
        D[counts_mask] = np.nan

    D[D >= 1-err_tol] = np.nan

    for i in range(M):
        D_i = np.copy(D[:,i])
        D_i = D_i.reshape(-1,1)
        D_i = np.repeat(D_i, M-1, 1)

        D_js = np.delete(D, i, 1)
        # Avoid div by 0
        D_js[D_js <= 0+err_tol] = np.nan
        if num_tol is not None:
            mask1 = np.logical_and(D_i >= 1-num_tol, D_js >= 1-num_tol)
            mask2 = np.logical_and(D_i <= 0+num_tol, D_js <= 0+num_tol)
            D_i[mask1] = np.nan
            D_i[mask2] = np.nan
            D_js[mask1] = np.nan
            D_js[mask2] = np.nan
        if log_tr:
            R_i = np.log(np.log(1-D_i)/np.log(1-D_js) + 1)
        else:
            R_i = np.log(1-D_i)/np.log(1-D_js)
        R_i = np.concatenate(R_i)
        R_i = R_i[~np.isnan(R_i)]
        if len(R_i) == 0:
            R_i = np.array([np.nan, np.nan])
        R.append(R_i)
    R = np.array(R)
    return R

def calc_rankj_values(D, counts=None, low_counts=None, err_tol=1e-5, num_tol=0.09):
    M = D.shape[1]
    N = D.shape[0]

    if counts is not None and low_counts is not None:
        counts_mask = counts < low_counts
        D[counts_mask] = np.nan

    # Avoid log(0)
    D[D >= 1-err_tol] = np.nan

    rankR = [np.array([]) for i in range(M)]
    for j in range(M):
        D_j = np.copy(D[:,j])
        # Avoid div by 0
        D_j[D_j <= 0+err_tol] = np.nan
        D_j = D_j.reshape(-1,1)
        D_j = np.repeat(D_j, M, 1)

        if num_tol is not None:
            mask1 = np.logical_and(D >= 1-num_tol, D_j >= 1-num_tol)
            mask2 = np.logical_and(D <= 0+num_tol, D_j <= 0+num_tol)
            D[mask1] = np.nan
            D[mask2] = np.nan
            D_j[mask1] = np.nan
            D_j[mask2] = np.nan

        R_j = np.log(1-D)/np.log(1-D_j)

        ranksj = np.argsort(np.argsort(-R_j, 1), 1) + 1
        ranksj[np.isnan(R_j)] = 0 # Mark unkown ranks

        for k in range(len(rankR)):
            rank_k = ranksj[:,k]
            rank_k = rank_k[rank_k!=0]
            rankR[k] = np.append(rankR[k], rank_k)
    rankR = np.array(rankR)
    return rankR

def calc_rankRj(D, counts=None, low_counts=None, sort='median', err_tol=1e-5, num_tol=0.09):
    M = D.shape[1]
    N = D.shape[0]

    if counts is not None and low_counts is not None:
        counts_mask = counts < low_counts
        D[counts_mask] = np.nan

    # Avoid log(0)
    D[D >= 1-err_tol] = np.nan

    if sort == 'median':
        m_func = np.nanmedian
    elif sort == 'mean':
        m_func = np.nanmean

    rankR = []
    for j in range(M):
        D_j = np.copy(D[:,j])
        # Avoid div by 0
        D_j[D_j <= 0+err_tol] = np.nan
        D_j = D_j.reshape(-1,1)
        D_j = np.repeat(D_j, M, 1)

        if num_tol is not None:
            mask1 = np.logical_and(D >= 1-num_tol, D_j >= 1-num_tol)
            mask2 = np.logical_and(D <= 0+num_tol, D_j <= 0+num_tol)
            D[mask1] = np.nan
            D[mask2] = np.nan
            D_j[mask1] = np.nan
            D_j[mask2] = np.nan

        R_j = np.log(1-D)/np.log(1-D_j)
        m_values = m_func(R_j, 0)
        rankj = np.argsort(np.argsort(m_values)) + 1
        rankR.append(rankj)
    return np.squeeze(stats.mode(rankR, 0).mode)

def Rlambda_distr(trps_vs_all, trps_data, sort_by='median', path=None,
                  base_name=None, log=False, lim_y=None, return_pl=False):

    num_dims = len(trps_vs_all)

    means = np.array([np.mean(v) for v in trps_vs_all])
    medians = np.array([np.median(v) for v in trps_vs_all])
    if sort_by == 'median':
        sort_idx = np.argsort(medians)
        medians = medians[sort_idx]
        means = means[sort_idx]
        trps_vs_all = trps_vs_all[sort_idx]
        trps_data = trps_data[sort_idx]
    elif sort_by == 'mean':
        sort_idx = np.argsort(means)
        medians = medians[sort_idx]
        means = means[sort_idx]
        trps_vs_all = trps_vs_all[sort_idx]
        trps_data = trps_data[sort_idx]
    elif sort_by == 'mode':
        modes = np.array([stats.mode(v).mode[0] for v in trps_vs_all])
        sort_idx = np.argsort(modes)
        medians = medians[sort_idx]
        means = means[sort_idx]
        modes = modes[sort_idx]
        trps_vs_all = trps_vs_all[sort_idx]
        trps_data = trps_data[sort_idx]

    x_labels = [trp + '\nvs.\nAll' for trp in trps_data['tripep']]
    quartile1 = [np.nanpercentile(v, 25) for v in trps_vs_all]
    quartile3 = [np.nanpercentile(v, 75) for v in trps_vs_all]

    fig = plt.figure(figsize=(20,10), dpi=300)
    ax = fig.add_subplot(111)
    parts = ax.violinplot(trps_vs_all.T, widths=0.7,
                          showextrema=False, showmedians=False,
                          showmeans=False, bw_method='scott')
    for pc in parts['bodies']:
        pc.set_facecolor('lightsteelblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.9)


    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(trps_vs_all, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='r', s=60, zorder=4,
               label='median')
    ax.scatter(inds, means, marker='D', color='g', s=60, zorder=4,
               label='mean')
    if sort_by == 'mode':
        xmins = [i-0.2 for i in inds]
        xmaxs = [i+0.2 for i in inds]
        ax.hlines(modes, xmins, xmaxs, color='grey',
                  linestyle='-', lw=1)
    ax.vlines(inds, quartile1, quartile3, color='grey',
              linestyle='-', lw=5, alpha=1)
    ax.vlines(inds, whiskersMin, whiskersMax, color='grey',
              linestyle='-', lw=1, alpha=1)
    ax.legend(fontsize='xx-large', loc=2,
              fancybox=True, facecolor='lightgrey')
    ax.set_xticks(inds)
    ax.set_xticklabels(x_labels, size='xx-large')
    # ax.set_xlim(0.25, len(trps_data['tripep']) + 0.75)
    if lim_y is not None:
        ax.set_ylim(0, lim_y)
    if log == True:
        ax.set_ylabel(r'$\log(k_A/k_B+1)$', size='xx-large', color='black')
    else:
        ax.set_ylabel(r'$k_A/\k_B$', size='xx-large', color='black')
    if path is not None and base_name is not None:
        plt.savefig(path+base_name+'_vs_all_violin.png', dpi=300, format=None)
        plt.close()
    else:
        if return_pl:
            return(fig)
        else:
            plt.show()



def Rlambda_scatter(lambda_pw3D, trps_data, lambdas,
                    path=None, base_name=None, **kwargs):
    num_dims = len(lambda_pw3D)
    trps_vs_all = []
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    inds = np.arange(1, len(lambdas)+1)
    for compared_with, trp, pos in zip(lambda_pw3D.T, trps_data['tripep'], inds):
        # print(np.repeat(pos,len(lambdas)))
        # ax.boxplot(compared_with, positions = np.repeat(pos,len(lambdas)))
        for vector, l in zip(compared_with, lambdas):
            ax.scatter(np.repeat(l,len(vector)), vector, alpha=0.5)
    plt.savefig(path+base_name+'_Rlambda_scatter.png', dpi=300, format=None)
    plt.close()
