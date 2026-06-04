# -*- coding: utf-8 -*-
"""
Created on Fri May 29 22:30:19 2026

@author: 18307
"""

import os
import numpy as np
import pandas as pd

import torch
import cnn_validation
from models import models

# %% Competing Methods
from utils import utils_feature_loading
def cnn_subnetworks_evaluation_circle_original_cm(feature_cm='pcc', normalization_for_train=False,
                                                  subject_range=range(6,16), experiment_range=range(1,4),
                                                  node_retention_list=None,
                                                  save=False):
    if node_retention_list is not None:
        channel_selects = {'gamma': node_retention_list,
                           'beta': node_retention_list,
                           'alpha': node_retention_list
                           }
        node_retention_number = len(node_retention_list)
    
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

            # Normalization before training
            if normalization_for_train:
                alpha = feature_engineering.normalize_matrix(alpha)
                beta = feature_engineering.normalize_matrix(beta)
                gamma = feature_engineering.normalize_matrix(gamma)

            # Selected CM           
            alpha_selected = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            
            # traning and testing
            result_CM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_CM}
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
    
    # Summary
    df_summary = pd.DataFrame([mean_row, std_row])
    summary_transpose = {
        "accuracy_avg": df_summary["accuracy"][0],
        "acc_std": df_summary["accuracy"][1],

        "f1_score_avg": df_summary["f1_score"][0],
        "f1_std": df_summary["f1_score"][1],

        "recall_avg": df_summary["recall"][0],
        "recall_std": df_summary["recall"][1],

        "loss_avg": df_summary["loss"][0],
        "loss_std": df_summary["loss"][1],
    }

    df_summary_transpose = pd.DataFrame([summary_transpose])
    
    # Save
    if save:        
        folder_name = 'results_(stress_test)_original'
        file_name = f'cnn_evaluation(stress_test)_{feature_cm}_origin.xlsx'
        sheet_name = f'nrn_{node_retention_number}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")  
        
    return df_results

from feature_engineering_slf import read_fcs as read_fcs_slf
def cnn_subnetworks_evaluation_circle_competing_signal_level(feature_cm='pcc', normalization_for_train=False,
                                                             method='Surface_Laplacian_Filtering', 
                                                             subject_range=range(6,16), experiment_range=range(1,4), 
                                                             node_retention_list=None,
                                                             save=False):
    if method.lower() == 'surface_laplacian_filtering':
        # _method = 'slf'
        folder_name='functional_connectivity_slfed'
    elif method.lower() == 'spatio_spectral_decomposition':
        # _method = 'ssd'
        folder_name='functional_connectivity_ssded'
        
    # subnetwork extraction----start
    if node_retention_list is not None:
        channel_selects = {'gamma': node_retention_list,
                           'beta': node_retention_list,
                           'alpha': node_retention_list
                           }
        node_retention_number = len(node_retention_list)
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
            
            # Normalization before training
            if normalization_for_train:
                alpha = feature_engineering.normalize_matrix(alpha)
                beta = feature_engineering.normalize_matrix(beta)
                gamma = feature_engineering.normalize_matrix(gamma)
            
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
    
    # Summary
    df_summary = pd.DataFrame([mean_row, std_row])
    summary_transpose = {
        "accuracy_avg": df_summary["accuracy"][0],
        "acc_std": df_summary["accuracy"][1],

        "f1_score_avg": df_summary["f1_score"][0],
        "f1_std": df_summary["f1_score"][1],

        "recall_avg": df_summary["recall"][0],
        "recall_std": df_summary["recall"][1],

        "loss_avg": df_summary["loss"][0],
        "loss_std": df_summary["loss"][1],
    }

    df_summary_transpose = pd.DataFrame([summary_transpose])

    # Save
    if save:        
        folder_name = 'results_(stress_test)_comps'
        file_name = f'cnn_evaluation(stress_test)_{feature_cm}_by_{method}.xlsx'
        sheet_name = f'nrn_{node_retention_number}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")  
        
    return df_results

import feature_engineering
from vce_model_fitting_competing import cm_rebuilding_competing
def cnn_subnetworks_evaluation_circle_competing_network_level(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                                                              feature_cm='pcc', normalization_for_train=False,
                                                              model='Generalized_Surface_Laplacian', model_fm='basic',
                                                              param='fitted_results_competing(sub1_sub5_joint_band)',
                                                              subject_range=range(6,16), experiment_range=range(1,4), 
                                                              node_retention_list=None,
                                                              save=False):
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_fm, method='competing', feature=feature_cm)
    global para
    para = param
    
    # subnetwork extraction----start
    if node_retention_list is not None:
        channel_selects = {'gamma': node_retention_list,
                           'beta': node_retention_list,
                           'alpha': node_retention_list
                           }
        node_retention_number = len(node_retention_list)
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
            
            # Normalization before training
            if normalization_for_train:
                alpha_rebuilded = feature_engineering.normalize_matrix(alpha_rebuilded)
                beta_rebuilded = feature_engineering.normalize_matrix(beta_rebuilded)
                gamma_rebuilded = feature_engineering.normalize_matrix(gamma_rebuilded)
            
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
    
    # Summary
    df_summary = pd.DataFrame([mean_row, std_row])
    summary_transpose = {
        "accuracy_avg": df_summary["accuracy"][0],
        "acc_std": df_summary["accuracy"][1],

        "f1_score_avg": df_summary["f1_score"][0],
        "f1_std": df_summary["f1_score"][1],

        "recall_avg": df_summary["recall"][0],
        "recall_std": df_summary["recall"][1],

        "loss_avg": df_summary["loss"][0],
        "loss_std": df_summary["loss"][1],
    }

    df_summary_transpose = pd.DataFrame([summary_transpose])

    # Save
    if save:        
        folder_name = 'results_(stress_test)_comps'
        file_name = f'cnn_evaluation(stress_test)_{feature_cm}_by_{model}.xlsx'
        sheet_name = f'nrn_{node_retention_number}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")  
        
    return df_results

# %% DM-RC; Proposed Method
from connectivity_matrix_rebuilding import cm_rebuilding as cm_rebuild
def cnn_subnetworks_evaluation_circle_DM_RC(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                                            feature_cm='pcc', normalization_for_train=False,
                                            model='Exponential', model_fm='basic', model_rcm='linear',
                                            subject_range=range(6,16), experiment_range=range(1,4), 
                                            node_retention_list=None,
                                            save=False):
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_fm, model_rcm, feature=feature_cm)
    
    # subnetwork extraction----start
    if node_retention_list is not None:
        channel_selects = {'gamma': node_retention_list,
                           'beta': node_retention_list,
                           'alpha': node_retention_list
                           }
        node_retention_number = len(node_retention_list)
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
            alpha_rebuilded = cm_rebuild(alpha, dm, param, model, model_fm, model_rcm, False, False)
            beta_rebuilded = cm_rebuild(beta, dm, param, model, model_fm, model_rcm, False, False)
            gamma_rebuilded = cm_rebuild(gamma, dm, param, model, model_fm, model_rcm, False, False)
            
            # Normalization before training
            if normalization_for_train:
                alpha_rebuilded = feature_engineering.normalize_matrix(alpha_rebuilded)
                beta_rebuilded = feature_engineering.normalize_matrix(beta_rebuilded)
                gamma_rebuilded = feature_engineering.normalize_matrix(gamma_rebuilded)
            
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
    
    # Summary
    df_summary = pd.DataFrame([mean_row, std_row])
    summary_transpose = {
        "accuracy_avg": df_summary["accuracy"][0],
        "acc_std": df_summary["accuracy"][1],

        "f1_score_avg": df_summary["f1_score"][0],
        "f1_std": df_summary["f1_score"][1],

        "recall_avg": df_summary["recall"][0],
        "recall_std": df_summary["recall"][1],

        "loss_avg": df_summary["loss"][0],
        "loss_std": df_summary["loss"][1],
    }

    df_summary_transpose = pd.DataFrame([summary_transpose])

    # Save
    if save:        
        folder_name = 'results_(stress_test)_dmrc'
        file_name = f'cnn_evaluation(stress_test)_{model}_{feature_cm}_({model_fm}_fm_{model_rcm}_rcm).xlsx'
        sheet_name = f'nrn_{node_retention_number}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        save_to_xlsx_sheet(df_summary, folder_name, file_name, "summary")
        save_to_xlsx_sheet(df_summary_transpose, folder_name, file_name, "summary_t")  
        
    return df_results

def cnn_subnetworks_eval_circle_rcm_intergrated(model_fm='basic', model_rcm='linear', 
                                                feature_cm='pcc', subject_range=range(6,16), 
                                                subnetworks_extract='separate_index', node_retention_list=None, save=False):
    model = list(['exponential', 'gaussian', 'inverse', 'power_law', 'rational_quadratic', 'generalized_gaussian', 'sigmoid'])
    model = list(['exponential', 'rational_quadratic', 'generalized_gaussian', 'sigmoid'])
    
    results_fitting = {}
    for trail in range(0, 4):
    #     # model_=model[trail]
    #     # print(model_)
    
        results_fitting[model[trail]] = cnn_subnetworks_evaluation_circle_DM_RC(
            projection_params={"source": "auto", "type": "3d_euclidean"}, 
            feature_cm=feature_cm, normalization_for_train=True,
            model=model[trail], model_fm=model_fm, model_rcm=model_rcm,
            subject_range=subject_range, experiment_range=range(1,4), 
            node_retention_list=node_retention_list,
            save=save)
    
    return results_fitting

# %% read parameters/save
def read_params(model='exponential', model_fm='basic', model_rcm='differ', 
                method='residual', feature='pcc'):
    if method == 'residual':
        identifier = f'dmrc,{model_fm.lower()}_fm_{model_rcm.lower()}_rcm,{feature}'
    elif method == 'competing':
        identifier = f'comps,{model_fm.lower()},{feature}'
    
    path_current = os.getcwd()
    path_fitting_results = os.path.join(path_current, 'parameters_optimized')
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
ch_index_62 = list(range(1,63))
ch_index_32 = [1,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,53,55,59,60,61]
ch_index_16 = [1,3,8,10,12,24,26,28,30,32,44,46,48,59,60,61]
ch_index_8 = [1,3,26,30,44,48,59,61]
ch_index_4 = [1,3,44,48]

ch_index_62 = [ch - 1 for ch in ch_index_62]
ch_index_32 = [ch - 1 for ch in ch_index_32]
ch_index_16 = [ch - 1 for ch in ch_index_16]
ch_index_8 = [ch - 1 for ch in ch_index_8]
ch_index_4 = [ch - 1 for ch in ch_index_4]

if __name__ == '__main__':
    # [ch_index_62, ch_index_32, ch_index_16, ch_index_8, ch_index_4]
    for _list in [ch_index_32, ch_index_16, ch_index_8, ch_index_4]:
        # # Baselines
        cnn_subnetworks_evaluation_circle_original_cm(feature_cm='pcc', normalization_for_train=True,
                                                      subject_range=range(6,16), experiment_range=range(1,4),
                                                      node_retention_list=_list, save=True)
        
        # DM-RCs
        cnn_subnetworks_eval_circle_rcm_intergrated(model_fm='basic', # 'basic'; 'advanced'
                                                    model_rcm='linear', # 'linear'; 'linear_ratio'
                                                    feature_cm='pcc', subject_range=range(1, 16),
                                                    node_retention_list=_list, save=True)
        
        # Competing; GSLF
        cnn_subnetworks_evaluation_circle_competing_network_level(feature_cm='pcc',normalization_for_train=True,
                                                                  model='Generalized_Surface_Laplacian', model_fm='basic', 
                                                                  # 'basic', 'advanced'
                                                                  subject_range=range(6,16),
                                                                  node_retention_list=_list, save=True)

        # competing; GLF
        cnn_subnetworks_evaluation_circle_competing_network_level(feature_cm='pcc',normalization_for_train=True,
                                                                  model='Graph_Laplacian_filtering', model_fm='basic', 
                                                                  # 'basic', 'advanced'
                                                                  subject_range=range(6,16),
                                                                  node_retention_list=_list, save=True)

        # competing; SLF
        cnn_subnetworks_evaluation_circle_competing_signal_level(feature_cm='pcc', normalization_for_train=True,
                                                                 method='Surface_Laplacian_Filtering', 
                                                                 subject_range=range(6,16), 
                                                                 node_retention_list=_list, save=True)

        # competing; SSD
        cnn_subnetworks_evaluation_circle_competing_signal_level(feature_cm='pcc', normalization_for_train=True,
                                                                 method='Spatio_Spectral_Decomposition', 
                                                                 subject_range=range(6,16), 
                                                                 node_retention_list=_list, save=True)
        
    # %% End
    from cnn_val_circle import end_program_actions
    end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)