# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:07:01 2026

@author: 18307
"""
import os
import numpy as np
import pandas as pd

import torch
import cnn_validation
from models import models
from utils import utils_feature_loading

# %% origin pcc
fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', 'plv', sub_range=range(1,6))
alpha_global_averaged = fcs_global_averaged['alpha']
beta_global_averaged = fcs_global_averaged['beta']
gamma_global_averaged = fcs_global_averaged['gamma']

strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=1)
strength_beta = np.sum(np.abs(beta_global_averaged), axis=1)
strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=1)

channel_weights = (strength_alpha+strength_beta+strength_gamma)/3


channels = utils_feature_loading.read_distribution('seed')
channel_weights_df = pd.DataFrame({"weights": channel_weights})
channels_df = pd.concat([channels, channel_weights_df], axis=1)

channels_sorted = channels_df.sort_values(by='weights')
channels_sorted_retained = channels_sorted.iloc[0:12, :]
channels_sorted_plv_retained = channels_sorted_retained.sort_values(by='order')

# %% dmrc
import feature_engineering
_, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
dm = feature_engineering.normalize_matrix(dm)

from cnn_subnetworks_val_circl_new import read_params
from connectivity_matrix_rebuilding import cm_rebuilding as cm_rebuild

model_fm='basic'
model_rcm='linear'

# exp
model='exponential'
param = read_params(model, model_fm, model_rcm, feature='pcc')

alpha_global_averaged_rebuilded = cm_rebuild(alpha_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
beta_global_averaged_rebuilded = cm_rebuild(beta_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
gamma_global_averaged_rebuilded = cm_rebuild(gamma_global_averaged, dm, param, model, model_fm, model_rcm, True, False)

strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)

channel_weights = (strength_alpha+strength_beta+strength_gamma)/3

channels = utils_feature_loading.read_distribution('seed')
channel_weights_df = pd.DataFrame({"weights": channel_weights})
channels_df = pd.concat([channels, channel_weights_df], axis=1)

channels_sorted = channels_df.sort_values(by='weights')
channels_sorted_retained = channels_sorted.iloc[0:12, :]
channels_sorted_dmrc_exp_retained = channels_sorted_retained.sort_values(by='order')

# rq
model='rational_quadratic'
param = read_params(model, model_fm, model_rcm, feature='pcc')

alpha_global_averaged_rebuilded = cm_rebuild(alpha_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
beta_global_averaged_rebuilded = cm_rebuild(beta_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
gamma_global_averaged_rebuilded = cm_rebuild(gamma_global_averaged, dm, param, model, model_fm, model_rcm, True, False)

strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)

channel_weights = (strength_alpha+strength_beta+strength_gamma)/3

channels = utils_feature_loading.read_distribution('seed')
channel_weights_df = pd.DataFrame({"weights": channel_weights})
channels_df = pd.concat([channels, channel_weights_df], axis=1)

channels_sorted = channels_df.sort_values(by='weights')
channels_sorted_retained = channels_sorted.iloc[0:12, :]
channels_sorted_dmrc_rq_retained = channels_sorted_retained.sort_values(by='order')

# gg
model='generalized_gaussian'
param = read_params(model, model_fm, model_rcm, feature='pcc')

alpha_global_averaged_rebuilded = cm_rebuild(alpha_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
beta_global_averaged_rebuilded = cm_rebuild(beta_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
gamma_global_averaged_rebuilded = cm_rebuild(gamma_global_averaged, dm, param, model, model_fm, model_rcm, True, False)

strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)

channel_weights = (strength_alpha+strength_beta+strength_gamma)/3

channels = utils_feature_loading.read_distribution('seed')
channel_weights_df = pd.DataFrame({"weights": channel_weights})
channels_df = pd.concat([channels, channel_weights_df], axis=1)

channels_sorted = channels_df.sort_values(by='weights')
channels_sorted_retained = channels_sorted.iloc[0:12, :]
channels_sorted_dmrc_gg_retained = channels_sorted_retained.sort_values(by='order')

# sig
model='sigmoid'
param = read_params(model, model_fm, model_rcm, feature='pcc')

alpha_global_averaged_rebuilded = cm_rebuild(alpha_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
beta_global_averaged_rebuilded = cm_rebuild(beta_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
gamma_global_averaged_rebuilded = cm_rebuild(gamma_global_averaged, dm, param, model, model_fm, model_rcm, True, False)

strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)

channel_weights = (strength_alpha+strength_beta+strength_gamma)/3

channels = utils_feature_loading.read_distribution('seed')
channel_weights_df = pd.DataFrame({"weights": channel_weights})
channels_df = pd.concat([channels, channel_weights_df], axis=1)

channels_sorted = channels_df.sort_values(by='weights')
channels_sorted_retained = channels_sorted.iloc[0:12, :]
channels_sorted_dmrc_sig_retained = channels_sorted_retained.sort_values(by='order')

# %% comps network
# from vce_model_fitting_competing import cm_rebuilding_competing

# model = 'generalized_surface_laplacian'
# param = read_params(model, model_fm, method='competing')

# alpha_global_averaged_rebuilded = cm_rebuilding_competing(alpha_global_averaged, dm, param, model)
# beta_global_averaged_rebuilded = cm_rebuilding_competing(beta_global_averaged, dm, param, model)
# gamma_global_averaged_rebuilded = cm_rebuilding_competing(gamma_global_averaged, dm, param, model)

# strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=0)
# strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=0)
# strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=0)

# channel_weights = (strength_alpha+strength_beta+strength_gamma)/3

# channels = utils_feature_loading.read_distribution('seed')
# channel_weights_df = pd.DataFrame({"weights": channel_weights})
# channels_df = pd.concat([channels, channel_weights_df], axis=1)

# channels_sorted = channels_df.sort_values(by='weights')
# channels_sorted_retained = channels_sorted.iloc[0:13, :]
# channels_sorted_glf_retained = channels_sorted_retained.sort_values(by='order')