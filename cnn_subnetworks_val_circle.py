# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:21:23 2025

@author: usouu
"""
import os
import numpy as np
import pandas as pd

import torch
import cnn_validation
from models import models

# %% Competing Methods
from utils import utils_feature_loading
def cnn_subnetworks_evaluation_circle_original_cm(selection_rate=1, feature_cm='pcc', 
                                                 subject_range=range(6,16), experiment_range=range(1,4), 
                                                 subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                                 save=False):
    if subnetworks_extract == 'read':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
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

    elif subnetworks_extract == 'calculation':
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        
        for sub in subnetworks_exrtact_basis:
            for ex in experiment_range:
                subject_id = f"sub{sub}ex{ex}"
                print(f"Evaluating {subject_id}...")
    
                # CM/H5
                features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
                alpha = features['alpha']
                beta = features['beta']
                gamma = features['gamma']
                
                # Compute node strength
                strength_alpha = np.sum(np.abs(alpha), axis=1)
                strength_beta = np.sum(np.abs(beta), axis=1)
                strength_gamma = np.sum(np.abs(gamma), axis=1)
                
                # Save for further analysis
                functional_node_strength['alpha'].append(strength_alpha)
                functional_node_strength['beta'].append(strength_beta)
                functional_node_strength['gamma'].append(strength_gamma)
    
        channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                           'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                           'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                           }
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
   
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Selected CM           
            alpha_selected = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:        
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_origin.xlsx'
        sheet_name = f'sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251220)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
        
    return df_results

from feature_engineering_slf import read_fcs as read_fcs_slf
from feature_engineering_slf import read_fcs_global_average as read_fcs_global_slf
def cnn_subnetworks_evaluation_circle_competing_signal_level(feature_cm='pcc', 
                                                            method='Surface_Laplacian_Filtering', 
                                                            subject_range=range(6,16), experiment_range=range(1,4), 
                                                            subnetworks_extract='separate_index', selection_rate=1,
                                                            subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                            save=False):
    # subnetwork extraction----start
    if method.lower() == 'surface_laplacian_filtering':
        # _method = 'slf'
        folder_name='functional_connectivity_slfed'
    elif method.lower() == 'spatio_spectral_decomposition':
        # _method = 'ssd'
        folder_name='functional_connectivity_ssded'
        
    if subnetworks_extract == 'unify_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global('seed', feature_cm, 'joint', subnets_exrtact_basis_sub)
        
    elif subnetworks_extract == 'separate_index':
        fcs_global_averaged = read_fcs_global_slf('seed', feature_cm, 'joint', subnets_exrtact_basis_sub, folder_name)
    
    strength_alpha = np.sum(np.abs(fcs_global_averaged['alpha']), axis=0)
    strength_beta = np.sum(np.abs(fcs_global_averaged['beta']), axis=0)
    strength_gamma = np.sum(np.abs(fcs_global_averaged['gamma']), axis=0)
    
    channel_importances = {'gamma': strength_gamma, 
                           'beta': strength_beta, 
                           'alpha': strength_alpha}
    
    k = {'gamma': int(len(channel_importances['gamma']) * selection_rate),
         'beta': int(len(channel_importances['beta']) * selection_rate),
         'alpha': int(len(channel_importances['alpha']) * selection_rate)}
    
    channel_selects = {'gamma': np.argsort(channel_importances['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_importances['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_importances['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetwork extraction----end
    
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # RCM/H5
            features = read_fcs_slf('seed', subject_id, feature_cm, folder_name=folder_name)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']
            
            # subnetworks
            alpha_rebuilded = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{method}_rcm.xlsx'
        sheet_name = f'GLF_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251220)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
        
    return df_results

import feature_engineering
from vce_model_fitting_competing import cm_rebuilding_competing
def cnn_subnetworks_evaluation_circle_competing_network_level(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                                                              feature_cm='pcc', 
                                                              model='Generalized_Surface_Laplacian', model_fm='basic',
                                                              param='fitted_results_competing(sub1_sub5_joint_band)',
                                                              subject_range=range(6,16), experiment_range=range(1,4), 
                                                              subnetworks_extract='separate_index', selection_rate=1,
                                                              subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                                              save=False):
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_fm, folder=param, method='competing')
    global para
    para = param
    
    # subnetwork extraction----start
    if subnetworks_extract == 'unify_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=0)
        strength_beta = np.sum(np.abs(beta_global_averaged), axis=0)
        strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=0)
        
        channel_importances = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
    elif subnetworks_extract == 'separate_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        alpha_global_averaged_rebuilded = cm_rebuilding_competing(alpha_global_averaged, dm, param, model)
        beta_global_averaged_rebuilded = cm_rebuilding_competing(beta_global_averaged, dm, param, model)
        gamma_global_averaged_rebuilded = cm_rebuilding_competing(gamma_global_averaged, dm, param, model)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=0)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=0)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=0)
        
        channel_importances = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}

    k = {'gamma': int(len(channel_importances['gamma']) * selection_rate),
         'beta': int(len(channel_importances['beta']) * selection_rate),
         'alpha': int(len(channel_importances['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_importances['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_importances['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_importances['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetwork extraction----end
    
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM
            alpha_rebuilded = cm_rebuilding_competing(alpha, dm, param, model)
            beta_rebuilded = cm_rebuilding_competing(beta, dm, param, model)
            gamma_rebuilded = cm_rebuilding_competing(gamma, dm, param, model)
            
            # subnetworks
            alpha_rebuilded = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{model}_rcm.xlsx'
        sheet_name = f'GLF_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251220)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')

    return df_results

# %% DM-RC; Proposed Method
from connectivity_matrix_rebuilding import cm_rebuilding as cm_rebuild
def cnn_subnetworks_evaluation_circle_DM_RC(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                                            feature_cm='pcc', 
                                            model='Exponential', model_fm='basic', model_rcm='linear',
                                            param='fitted_results(sub1_sub5_joint_band)', 
                                            subject_range=range(6,16), experiment_range=range(1,4), 
                                            subnetworks_extract='separate_index', selection_rate=1,
                                            subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
                                            save=False):
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_fm, model_rcm, folder=param)
    
    # subnetwork extraction----start
    if subnetworks_extract == 'unify_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=0)
        strength_beta = np.sum(np.abs(beta_global_averaged), axis=0)
        strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=0)
        
        channel_importances = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
    elif subnetworks_extract == 'separate_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, 'joint',
                                                                            subnets_exrtact_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        alpha_global_averaged_rebuilded = cm_rebuild(alpha_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
        beta_global_averaged_rebuilded = cm_rebuild(beta_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
        gamma_global_averaged_rebuilded = cm_rebuild(gamma_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=0)
        strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=0)
        strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=0)
        
        channel_importances = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}

    k = {'gamma': int(len(channel_importances['gamma']) * selection_rate),
         'beta': int(len(channel_importances['beta']) * selection_rate),
         'alpha': int(len(channel_importances['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_importances['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_importances['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_importances['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetwork extraction----end
    
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # RCM
            alpha_rebuilded = cm_rebuild(alpha, dm, param, model, model_fm, model_rcm, True, False)
            beta_rebuilded = cm_rebuild(beta, dm, param, model, model_fm, model_rcm, True, False)
            gamma_rebuilded = cm_rebuild(gamma, dm, param, model, model_fm, model_rcm, True, False)
            
            # subnetworks
            alpha_rebuilded = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_rebuilded = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_rebuilded = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_rebuilded = np.stack((alpha_rebuilded, beta_rebuilded, gamma_rebuilded), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            # traning and testing
            result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuilded, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:
        folder_name = 'results_cnn_subnetwork_evaluation'
        file_name = f'cnn_validation_SubRCM_{feature_cm}_by_{model_fm}_fm_{model_rcm}_rcm.xlsx'
        sheet_name = f'{model}_sr_{selection_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary (20251220)
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')

    return df_results

def cnn_subnetworks_eval_circle_rcm_intergrated(model_fm='basic', model_rcm='linear', 
                                                feature_cm='pcc', subject_range=range(6,16), 
                                                subnetworks_extract='separate_index', selection_rate=1, save=False):
    model = list(['exponential', 'gaussian', 'inverse', 'power_law', 'rational_quadratic', 'generalized_gaussian', 'sigmoid'])
    
    results_fitting = {}
    for trail in range(0, 7):       
        results_fitting[model[trail]] = cnn_subnetworks_evaluation_circle_DM_RC(
            projection_params={"source": "auto", "type": "3d_euclidean"}, 
            feature_cm=feature_cm, 
            model=model[trail], model_fm=model_fm, model_rcm=model_rcm,
            param='fitted_results(sub1_sub5_joint_band)', 
            subject_range=subject_range, experiment_range=range(1,4), 
            subnetworks_extract=subnetworks_extract, selection_rate=selection_rate,
            subnets_exrtact_basis_sub=range(1,6), subnets_exrtact_basis_ex=range(1,4),
            save=save)
    
    return results_fitting

# %% read parameters/save
def read_params(model='exponential', model_fm='basic', model_rcm='differ', folder='fitted_results(sub1_sub5_joint_band)', method='residual'):
    if method == 'residual':
        identifier = f'{model_fm.lower()}_fm_{model_rcm.lower()}_rcm'
    elif method == 'competing':
        identifier = f'{model_fm.lower()}'
    
    path_current = os.getcwd()
    path_fitting_results = os.path.join(path_current, 'fitted_results', folder)
    file_path = os.path.join(path_fitting_results, f'fitted_results({identifier}).xlsx')
    
    df = pd.read_excel(file_path).set_index('method')
    df_dict = df.to_dict(orient='index')
    
    model = model.upper()
    params = df_dict[model]
    
    return params

def save_to_xlsx_sheet(df, folder_name, file_name, sheet_name):
    output_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)

    # if file exsist
    if os.path.exists(file_path):
        try:
            # try to read sheet
            existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
        except ValueError:
            # if sheet not exsist then create empty DataFrame
            existing_df = pd.DataFrame()

        # concat by column
        df = pd.concat([existing_df, df], ignore_index=True)

        # continuation + replace
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # if file not exsist then create
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    
def save_to_xlsx_fitting(results, subject_range, experiment_range, folder_name, file_name, sheet_name):
    # calculate average
    result_keys = results[0].keys()
    avg_results = {key: np.mean([res[key] for res in results]) for key in result_keys}
    
    # save to xlsx
    # 准备结果数据
    df_results = pd.DataFrame(results)
    df_results.insert(0, "Subject-Experiment", [f'sub{i}ex{j}' for i in subject_range for j in experiment_range])
    df_results.loc["Average"] = ["Average"] + list(avg_results.values())
    
    # 构造保存路径
    path_save = os.path.join(os.getcwd(), folder_name, file_name)
    
    # 判断文件是否存在
    if os.path.exists(path_save):
        # 追加模式，保留已有 sheet，添加新 sheet
        with pd.ExcelWriter(path_save, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 新建文件
        with pd.ExcelWriter(path_save, engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)

# %% Execute
if __name__ == '__main__':
    selection_rate_list = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    for selection_rate in selection_rate_list:
        # DM-RCs
        # cnn_subnetworks_eval_circle_rcm_intergrated(model_fm='advanced', # 'basic'; 'advanced'
        #                                             model_rcm='linear_ratio', # 'linear'; 'linear_ratio'
        #                                             feature_cm='plv', subject_range=range(6, 16), 
        #                                             subnetworks_extract='separate_index',
        #                                             selection_rate=selection_rate, save=True)
        
        # Competing; GSLF
        # cnn_subnetworks_evaluation_circle_competing_network_level(feature_cm='plv',
        #                                                           model='Generalized_Surface_Laplacian', model_fm='basic', 
        #                                                           # 'basic', 'advanced'
        #                                                           param='fitted_results_competing(sub1_sub5_joint_band)_kernel_norm',
        #                                                           subject_range=range(6,16),
        #                                                           subnetworks_extract='separate_index',
        #                                                           selection_rate=selection_rate, save=True)
        
        # competing; GLF
        # cnn_subnetworks_evaluation_circle_competing_network_level(feature_cm='plv',
        #                                                           model='Graph_Laplacian_filtering', model_fm='basic', 
        #                                                           # 'basic', 'advanced'
        #                                                           param='fitted_results_competing(sub1_sub5_joint_band)_kernel_norm',
        #                                                           subject_range=range(6,16),
        #                                                           subnetworks_extract='separate_index',
        #                                                           selection_rate=selection_rate, save=True)
        
        # competing; SLF
        # cnn_subnetworks_evaluation_circle_competing_signal_level(feature_cm='plv', 
        #                                                          method='Surface_Laplacian_Filtering', 
        #                                                          subject_range=range(6,16), 
        #                                                          subnetworks_extract='separate_index',
        #                                                          selection_rate=selection_rate, save=True)
        
        # competing; SSD
        cnn_subnetworks_evaluation_circle_competing_signal_level(feature_cm='plv', 
                                                                 method='Spatio_Spectral_Decomposition', 
                                                                 subject_range=range(6,16), 
                                                                 subnetworks_extract='separate_index',
                                                                 selection_rate=selection_rate, save=True)
        
    # %% End
    from cnn_val_circle import end_program_actions
    end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)