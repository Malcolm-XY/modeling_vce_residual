# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 22:18:11 2025

@author: 18307
"""

import numpy as np

import feature_engineering
from utils import utils_feature_loading
from feature_engineering_slf import read_fcs_global_average as read_fcs_global_slf
from vce_model_fitting_competing import cm_rebuilding_competing
from cnn_subnetworks_val_circle import read_params
def retrieve_node_list(method, feature_cm='pcc', selection_rate=1, subnetworks_exrtact_basis=range(1,6)):
    method = method.lower()
    if method == 'original':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
    elif method == 'surface_laplacian_filtering':
        fcs_global_averaged = read_fcs_global_slf('seed', feature_cm, 'joint', subnetworks_exrtact_basis, 'functional_connectivity_slfed')
    elif method == 'spatio_spectral_decomposition':
        fcs_global_averaged = read_fcs_global_slf('seed', feature_cm, 'joint', subnetworks_exrtact_basis, 'functional_connectivity_ssded')
    elif method == 'generalized_laplacian_filtering' or method == 'graph_laplacian_filtering':
        # parameters for construction of FM and RCM
        model, model_fm, folder = method, 'basic', 'fitted_results_competing(sub1_sub5_joint_band)'
        param = read_params(model=model, model_fm=model_fm, folder=folder, method='competing')

        # distance matrix
        _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
        dm = feature_engineering.normalize_matrix(dm)
        
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint', subnetworks_exrtact_basis)
        
        fcs_global_averaged['alpha'] = cm_rebuilding_competing(fcs_global_averaged['alpha'], dm, param, method)
        fcs_global_averaged['beta'] = cm_rebuilding_competing(fcs_global_averaged['beta'], dm, param, method)
        fcs_global_averaged['gamma'] = cm_rebuilding_competing(fcs_global_averaged['gamma'], dm, param, method)
    
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=1)
    strength_beta = np.sum(np.abs(beta_global_averaged), axis=1)
    strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=1)

    channel_weights = {'alpha': strength_alpha, 
                       'beta': strength_beta,
                       'gamma': strength_gamma,
                       }

    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }

    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }

    return channel_selects

selection_rate = 0.5

node_list_ori_alpha = retrieve_node_list('original', selection_rate)['alpha']
node_list_ori_beta = retrieve_node_list('original', selection_rate)['beta']
node_list_ori_gamma = retrieve_node_list('original', selection_rate)['gamma']

node_list_slf_alpha = retrieve_node_list('surface_laplacian_filtering', selection_rate)['alpha']
node_list_slf_beta = retrieve_node_list('surface_laplacian_filtering', selection_rate)['beta']
node_list_slf_gamma = retrieve_node_list('surface_laplacian_filtering', selection_rate)['gamma']

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

x = np.array([node_list_ori_alpha, node_list_ori_beta, node_list_ori_gamma,
              node_list_slf_alpha, node_list_slf_beta, node_list_slf_gamma])
x_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(x)

plt.scatter(x_embedded[:,0], x_embedded[:,1], cmap='viridis')

def calculate_overlap(list1, list2):
    set1, set2 = set(list1), set(set(list2))
    intersection = len(set1.intersection(set2))
    return intersection / len(set1)  # 因为长度固定，Jaccard或简单重合率均可

# 示例计算
overlap_alpha = calculate_overlap(node_list_ori_alpha, node_list_slf_alpha)
overlap_beta  = calculate_overlap(node_list_ori_beta, node_list_slf_beta)
overlap_gamma = calculate_overlap(node_list_ori_gamma, node_list_slf_gamma)