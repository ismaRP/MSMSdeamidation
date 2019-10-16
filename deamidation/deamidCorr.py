import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from sklearn import linear_model
from scipy.optimize import least_squares
import networkx as nx


# General options
plt.style.use('seaborn-dark-palette')



def pearson_corr(x,y):
    corr, pval = sp.stats.pearsonr(x,y)

    if pval > 0.1 or np.isnan(pval) or np.isnan(corr) or len(x) < 10:
        corr = np.nan
    return(corr)


def correlation(Ddata, trps_data, path, base_name):
    num_dims = Ddata.shape[1]
    corr_mat = np.zeros((num_dims, num_dims))
    for i in range(num_dims):
        for j in range(num_dims):
            D = Ddata[:, [j,i]]
            not_miss = ~np.isnan(D)
            keep = np.sum(not_miss, 1) == 2
            D = D[keep,:]
            x = D[:,0]#.reshape(-1,1)
            y = D[:,1]
            # Calculate correlation
            corr = pearson_corr(x,y)
            corr_mat[j,i] = corr

    str_trps = ','.join(trps_data['tripep'])
    out_file = path + base_name + '_correlation.csv'
    with open(out_file, 'w') as f:
        np.savetxt(f, corr_mat, fmt='%.2f', header=str_trps, delimiter=',', comments='')
    return corr_mat

def show_graph_with_labels(corr_mat, th, trps, halftimes, prop=None,
                           path=None, base_name=None):

    # Get a 0-1 matrix for whether corr > threshold
    adjacency_matrix = np.zeros(corr_mat.shape)
    adjacency_matrix[corr_mat >= th] = 1
    adjacency_matrix[corr_mat < th] = 0

    rows, cols = np.where(adjacency_matrix == 1)
    rows = rows.tolist()
    cols = cols.tolist()
    edges = []
    # keysm.set_array([])
    for i in range(len(rows)):
        d = {'corr': corr_mat[rows[i], cols[i]]}
        e = (trps[rows[i]], trps[cols[i]], d)
        edges.append(e)

    gr = nx.Graph()
    gr.add_edges_from(edges)

    edges, corrs = zip(*nx.get_edge_attributes(gr,'corr').items())
    corrs = np.array(corrs)
    if prop == None:
        pos = nx.kamada_kawai_layout(gr, weight='corr')
    else:
        pos = {}
        for node in gr.nodes():
            pos[node] = (halftimes[node]['X'][prop],
                         halftimes[node]['Y'][prop])
    hts = [halftimes[node]['lambda'] for node in gr.nodes()]
    hts = np.arcsinh(1*hts)/1
    node_cmap = plt.cm.coolwarm_r
    nodenorm = plt.Normalize(np.min(hts), np.max(hts))
    nodesm = ScalarMappable(norm=nodenorm, cmap=node_cmap)
    nodesm.set_array([])

    low_lim = np.min(corrs)
    max_lim = np.max(corrs[corrs<1.0])
    edge_cmap = plt.cm.Greys
    edgenorm = plt.Normalize(low_lim, max_lim)
    edgesm = ScalarMappable(norm=edgenorm, cmap=edge_cmap)
    edgesm.set_array([])

    fig = plt.figure(figsize=[10,6])
    nx.draw_networkx(gr, node_size=100, with_labels=True, pos=pos, alpha=0.9,
                     node_color=hts, cmap=node_cmap,
                     vmin=np.min(hts), vmax=np.max(hts),
                     edge_color=corrs, edge_cmap=edge_cmap,
                     edge_vmin=low_lim, edge_vmax=max_lim,
                     width=1.2, linewidths=2, font_size=5,
                     font_color='black', font_weight='semibold')
    corr_cbar = plt.colorbar(edgesm, orientation="vertical")
    corr_cbar.ax.set_title('Correlation')

    ht_cbar = plt.colorbar(nodesm, orientation='vertical')
    ht_cbar.ax.set_title('Reaction constant')

    plt.title('Correlation between tripeptides')
    if prop != None:
        plt.xlabel('X ' + prop)
        plt.ylabel('Z ' + prop)

    if path != None and base_name != None:
        out_file = path + base_name + '_' + prop + '_graph.png'
        plt.savefig(out_file, dpi = 400)
    else:
        plt.show()
    return(gr)


# Non-linear regression funcionts
def lsf1par(a, x, y):
    return np.exp(a*(x-1)) - y

def lsf1par_inv(a, x, y):
    x_nonz = x[x!=0]
    log_x_nonz = np.log(x_nonz)
    log_x = np.zeros(x.shape[0])
    log_x[x!=0] = log_x_nonz

    return log_x/a + 1 - y

def lsf2par(b, x, y):
    return np.exp(b[0] + b[1]*x) - y


def export_legend(legend, key = None, expand=[-5,-5,5,5]):
    filename = "out/legend_{}.png".format(key)
    fig = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=bbox)


def transform(v, t, theta=1):
    """
    Apply a transformation to the data in order to improve the
    visualization
    Input:
        -v: numeric vector with the data to be transformed
        -t: transformation to be applied.
            "log": logarithmic transofrmatio,
            "ihs": inverse hyperbolic sine transformation. In this case there
                is a theta parameter associated
            "norm": v-mu/sigma2 produces 0 mean and unit variance
    """
    if t == 'log':
        v = np.log(v)
    elif t== 'ihs':
        v = np.arcsinh(theta*v)/theta
    elif t== 'norm':
        m = np.mean(v)
        std = np.std(v)
        v = (v-m)/std
    return(v)


def multiscatter(deamid_mat, key=None, type=None, hts=None,
                 t=None, path=None, base_name=None, low_counts=None,
                 **kwargs):
    """
    Creates tripetides multiscatter plot
    Input:
        - key: Ydata entry info to include in the plot
        - type: wether key is categorical ('cat') or continuous ('cont')
        - path: path in which plot is saved
    *Notes:
        - key can refer to numeric but still categorical data
    """

    Ddata = deamid_mat.D
    counts = deamid_mat.counts
    Ydata = deamid_mat.Ydata
    trps_data = deamid_mat.trps_data

    if key is not None:
        Yvect = Ydata[key]
    else:
        Yvect = np.zeros(Ddata.shape[0])

    if len(hts) == 0:
        exists_hts = False
    else:
        exists_hts = True

    # Detect samples with known Ydata
    if type == 'cat':
        known = Yvect != 'U'
    elif type == 'cont':
        known = Yvect != -1
    else:
        known = np.array([True for i in range(len(Yvect))])

    # Apply transformation to continuous data in order to visualize better
    if type == 'cont' and t != None:
        Yvect = transform(Yvect, t)

    # Include visual options from vopts to defaults
    dfts = {
        'fontsize': 12,
        'fontpos': [0.15, 0.35],
        'fontweight': 'medium',
        'mcolor': 'black',
        'msize': 5,
        'reg': False,
        'cat_cmap': 'tab10',
        'bbox_to_anchor': (0.5,2.2)
    }
    dfts.update(kwargs)

    # Create mapping for either continuous or categorical data
    if type == 'cat':
        Yset = np.sort(list(set(Yvect))) # Set (array type) of Y values
        keyCm = plt.get_cmap(dfts['cat_cmap'])
        keyNorm  = plt.Normalize(vmin=0, vmax=len(Yset))
        keySm = ScalarMappable(norm=keyNorm, cmap=keyCm)
        keySm.set_array([])
        # Handles for the legned
        handles = [plt.plot([], [], markersize=18, marker='.',
                   ls='none', c = keySm.to_rgba(v))[0] for v in range(len(Yset))]
        plt.close()

        map_color = np.zeros(Yvect.shape[0])
        i = 0
        for Yval in Yset:
            map_color[Yvect == Yval] = i
            i += 1
        Yvect = map_color
    elif type == 'cont':
        # Create mappable for colorbar for ages
        keyCm = plt.get_cmap("cool")
        keyNorm = plt.Normalize(np.min(Yvect[known]), np.max(Yvect[known]))
        keySm = ScalarMappable(norm=keyNorm, cmap=keyCm)
        keySm.set_array([])
    else:
        keyCm = plt.get_cmap("brg")
        keyNorm = plt.Normalize(0, 1)
        keySm = ScalarMappable(norm=keyNorm, cmap=keyCm)
        keySm.set_array([])


    # Create map for halftime
    if exists_hts == True:
        htcm = plt.get_cmap("coolwarm")
        htnorm = plt.Normalize(np.min(hts), np.max(hts))
        htsm = ScalarMappable(norm=htnorm, cmap=htcm)
        htsm.set_array([])

    num_dims = Ddata.shape[1]
    fig = plt.figure(figsize=(16, 12))
    axes = [[False for i in range(num_dims)] for j in range(num_dims)]
    n=1
    # Regressor
    # reg = linear_model.LinearRegression(fit_intercept=False)
    # ridge = linear_model.Ridge(alpha=0.1)
    for i in range(num_dims):
        for j in range(num_dims):
            ax = fig.add_subplot(num_dims, num_dims, n)
            # Extract j and i columns
            plot_data = Ddata[:, [j,i]]
            # Check for pairwise low counts data rowwise
            if low_counts is not None:
                counts_data = counts[:, [j,i]]
                not_miss = counts_data > low_counts
            else:
                not_miss = ~np.isnan(plot_data)
            sele = np.sum(not_miss, 1) == 2
            sele = np.array(sele)

            X = plot_data[:, 0]
            Y = plot_data[:, 1]
            # Yplot = Yvect[sele]
            sele_known = np.logical_and(sele, known)
            sele_unknown = np.logical_and(sele, ~known)
            if i != j:
                # Fit linear regression
                if X.shape[0]>1 and dfts['reg']==True:
                    x_pred = np.linspace(-0.1,1.1,500)
                    # Linear 1to1
                    # reg.fit(X.reshape(-1,1), Y)
                    # y_pred = reg.predict(x_pred.reshape(-1,1))
                    # pl = ax.plot(x_pred.reshape(-1,1),y_pred, color='lime', linewidth=1)
                    # pl = ax.plot([0,1], [0,1], color='lime', linewidth=0.6, alpha=0.8)

                    # Non-linear regression 1 parameter a
                    a0 = 1
                    res1 = least_squares(lsf1par, a0, loss='soft_l1', f_scale=0.1, args=(X,Y))
                    a = res1.x
                    y_pred = np.exp(a*(x_pred-1))
                    pl = ax.plot(x_pred, y_pred, color='black', linewidth=1)

                    # Non-linear regression 1 parameter a
                    a0 = 1
                    res2 = least_squares(lsf1par_inv, a0, loss='soft_l1', f_scale=0.1, args=(X,Y))
                    a = res2.x
                    x_pred1 =  np.linspace(0.0001,1.1,500)
                    y_pred = np.log(x_pred1)/a + 1
                    pl = ax.plot(x_pred1, y_pred, color='red', linewidth=1)

                if type != None:
                    sc = ax.scatter(X[sele_known], Y[sele_known],
                                    s=dfts['msize'], alpha=0.7,
                                    c=keySm.to_rgba(Yvect[sele_known]))
                    sc = ax.scatter(X[sele_unknown], Y[sele_unknown],
                                    s=dfts['msize'], alpha=0.7,
                                    c='black')
                else:
                    # rel = X/Y
                    # rel = np.logical_and(rel>0.8, rel<1.2)
                    # rel = rel*1
                    # sc = ax.scatter(X, Y, s=dfts['msize'], c=keySm.to_rgba(rel), alpha=0.8)
                    sc = ax.scatter(X[sele_known], Y[sele_known],
                                    s=dfts['msize'], c=dfts['mcolor'],
                                    alpha=0.8)
            elif i == j:
                txt = trps_data[i]['tripep']
                if trps_data[i][1] != 'NA':
                    txt = trps_data[i][0]+'\n'+txt

                if trps_data[i][4] != 0:
                    txt = txt + str(trps_data[i][4])
                elif trps_data[i][3] != 0:
                    txt = txt + str(trps_data[i][3])
                ax.text(dfts['fontpos'][0], dfts['fontpos'][1],
                        txt, fontweight=dfts['fontweight'],
                        fontsize=dfts['fontsize'])
                if exists_hts == False:
                    ax.set_facecolor('white')
                elif exists_hts == True and hts[i] != -1:
                    ax.set_facecolor(htsm.to_rgba(hts[i]))
                else:
                    ax.set_facecolor('white')
            # Set axis labels:
            ax.set_xlabel(trps_data['tripep'][j])
            ax.set_ylabel(trps_data['tripep'][i])
            # Equal scale axis
            ax.axis(xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=1.1)
            # Hide axes for all but the plots on the edge:
            if i < num_dims - 1:
                ax.xaxis.set_visible(False)
            if j > 0:
                ax.yaxis.set_visible(False)
            # Add this axis to the list.
            axes[j][i] = ax
            n += 1
    axes=np.array(axes)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1, wspace=0.2, hspace=0.2)
    # halftime color bar
    if exists_hts != False:
        htcbar_ax = fig.add_axes([0.92, 0.51, 0.015, 0.38])
        htcbar = plt.colorbar(htsm, cax=htcbar_ax, orientation="vertical")
        htcbar.ax.set_title("Reaction\nconstant")

    if type == 'cont': # Make a colorbar for it
        keycb_ax = fig.add_axes([0.92, 0.1, 0.015, 0.38])
        keycbar = plt.colorbar(keysm, cax=keycb_ax, orientation="vertical")
        if t != None:
            keycbar.ax.set_title(key+' ({})'.format(t))
        else:
            keycbar.ax.set_title(key)
    elif type == 'cat': # Set the legend for it
        # ax_pos =int(np.floor(num_dims/2))
        # axes[ax_pos,0].legend(handles, Yset,ncol=3,
        #                  bbox_to_anchor=dfts['bbox_to_anchor'],
        #                  fontsize='large')
        leg = plt.legend(handles, Yset,ncol=3,
                         bbox_to_anchor=dfts['bbox_to_anchor'],
                         fontsize='large')
        export_legend(leg, key)
    # Show or save
    if path is None and base_name is None:
        plt.show()
    else:
        plt.savefig(path+base_name+'_multiscatter.png', dpi=300, format=None)
        plt.close()
