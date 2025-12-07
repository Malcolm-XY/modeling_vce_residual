# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:32:18 2025

@author: 18307
"""
import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score

# %% svm foundation
def train_and_evaluate_svm(X_train, Y_train, X_val, Y_val):
    model = SVC(kernel='rbf', C=1, gamma='scale')
    model.fit(X_train, Y_train)
    
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(Y_val, val_preds) * 100
    recall = recall_score(Y_val, val_preds, average='weighted') * 100
    f1 = f1_score(Y_val, val_preds, average='weighted') * 100

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1
    }

def train_and_evaluate_knn(X_train, Y_train, X_val, Y_val, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)

    val_preds = model.predict(X_val)
    accuracy = accuracy_score(Y_val, val_preds) * 100
    recall = recall_score(Y_val, val_preds, average='weighted') * 100
    f1 = f1_score(Y_val, val_preds, average='weighted') * 100

    return {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1
    }

def k_fold_cross_validation_ml(X, Y, k_folds=5, use_sequential_split=True, model_type='svm', n_neighbors=5):
    X = np.array(X)
    Y = np.array(Y)

    results = []

    if use_sequential_split:
        fold_size = len(X) // k_folds
        indices = list(range(len(X)))

        for fold in range(k_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < k_folds - 1 else len(X)
            val_idx = indices[val_start:val_end]
            train_idx = indices[:val_start] + indices[val_end:]

            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            if model_type == 'svm':
                result = train_and_evaluate_svm(X_train, Y_train, X_val, Y_val)
            elif model_type == 'knn':
                result = train_and_evaluate_knn(X_train, Y_train, X_val, Y_val, n_neighbors=n_neighbors)

            results.append(result)

    else:
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            if model_type == 'svm':
                result = train_and_evaluate_svm(X_train, Y_train, X_val, Y_val)
            elif model_type == 'knn':
                result = train_and_evaluate_knn(X_train, Y_train, X_val, Y_val, n_neighbors=n_neighbors)

            results.append(result)

    avg_results = {
        'accuracy': np.mean([res['accuracy'] for res in results]),
        'recall': np.mean([res['recall'] for res in results]),
        'f1_score': np.mean([res['f1_score'] for res in results]),
    }

    print(f"{k_folds}-Fold Cross Validation Results ({model_type.upper()}):")
    print(f"Average Accuracy: {avg_results['accuracy']:.2f}%")
    print(f"Average Recall: {avg_results['recall']:.2f}%")
    print(f"Average F1 Score: {avg_results['f1_score']:.2f}%\n")

    return avg_results

# %% example usage
def example_usage():
    # Example Usage
    # Replace these with your actual data
    X_dummy = np.random.rand(100, 10)  # Example feature data
    Y_dummy = np.random.randint(0, 3, size=100)  # Example labels
    
    # SVM Evaluation
    svm_results = k_fold_cross_validation_ml(X_dummy, Y_dummy, k_folds=5, model_type='svm')
    
    # KNN Evaluation
    knn_results = k_fold_cross_validation_ml(X_dummy, Y_dummy, k_folds=5, model_type='knn', n_neighbors=5)
    
    # Save Results to Excel
    results = pd.DataFrame([svm_results, knn_results], index=['SVM', 'KNN'])
    output_path = os.path.join(os.getcwd(), 'Results', 'svm_knn_comparison.xlsx')
    results.to_excel(output_path, index=True, sheet_name='Comparison Results')

# %% evaluations
def svm_eval_circle_original_fn(selection_rate=1, feature_cm='pcc', 
                                subject_range=range(6,16), experiment_range=range(1,4), 
                                subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                                save=False):
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    labels = np.reshape(labels, -1)
    
    # features
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
    
    k = {'gamma': int(len(channel_weights['gamma']) * selection_rate),
         'beta': int(len(channel_weights['beta']) * selection_rate),
         'alpha': int(len(channel_weights['alpha']) * selection_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
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
            
            global x_selected
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            x_selected = x_selected.reshape(x_selected.shape[0], -1)
    
            # svm evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, labels, k_folds=5, model_type='svm')
            
            all_results_list.append(svm_results)
    
    print('Evaluation compelete\n')
    
    # calculate average
    result_keys = all_results_list[0].keys()
    avg_results = {key: np.mean([res[key] for res in all_results_list]) for key in result_keys}
    print(f'Average SVM Results: {avg_results}')
    
    # save to xlsx
    if save:
        save_to_xlsx(all_results_list, subject_range, experiment_range,
                             folder_name='results_svm_channel_selection_evaluation',
                             file_name=f'svm_validation_{feature_cm}_by_original_fn.xlsx',
                             sheet_name=f'selection_rate_{selection_rate}')
    
    return all_results_list, avg_results

import feature_engineering
from utils import utils_feature_loading
from connectivity_matrix_rebuilding import cm_rebuilding_competing
def svm_eval_circle_GSLF(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                         feature_cm='pcc', 
                         model='Generalized_Surface_Laplacian', model_config='basic', 
                         param='fitted_results_competing(sub1_sub5_joint_band)', kernel_normalization=False,
                         subject_range=range(6,16), experiment_range=range(1,4), selection_rate=1,
                         subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
                         save=False):
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_config, folder=param, method='competing')
    
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    labels = np.reshape(labels, -1)
    
    # features
    fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    alpha_global_averaged_rebuilded = cm_rebuilding_competing(alpha_global_averaged, dm, param, kernel_normalization)
    beta_global_averaged_rebuilded = cm_rebuilding_competing(beta_global_averaged, dm, param, kernel_normalization)
    gamma_global_averaged_rebuilded = cm_rebuilding_competing(gamma_global_averaged, dm, param, kernel_normalization)
    
    strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
    strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
    strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)
    
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
            alpha_rebuilded = cm_rebuilding_competing(alpha, dm, param, kernel_normalization)
            beta_rebuilded = cm_rebuilding_competing(beta, dm, param, kernel_normalization)
            gamma_rebuilded = cm_rebuilding_competing(gamma, dm, param, kernel_normalization)
            
            # Selected CM           
            alpha_selected = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            global x_selected
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            x_selected = x_selected.reshape(x_selected.shape[0], -1)
    
            # svm evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, labels, k_folds=5, model_type='svm')
            
            all_results_list.append(svm_results)
    
    print('Evaluation compelete\n')
    
    # calculate average
    result_keys = all_results_list[0].keys()
    avg_results = {key: np.mean([res[key] for res in all_results_list]) for key in result_keys}
    print(f'Average SVM Results: {avg_results}')
    
    # save to xlsx
    if save:
        save_to_xlsx(all_results_list, subject_range, experiment_range,
                             folder_name='results_svm_channel_selection_evaluation',
                             file_name=f'svm_validation_{feature_cm}_by_GSLF_{model_config}.xlsx',
                             sheet_name=f'selection_rate_{selection_rate}')
    
    return all_results_list, avg_results

from connectivity_matrix_rebuilding import cm_rebuilding as cm_rebuild
def svm_eval_circle_dmrc(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                         feature_cm='pcc', 
                         model='exponential', model_fm='basic', model_rcm='linear',
                         param='fitted_results(sub1_sub5_joint_band)', 
                         subject_range=range(6,16), experiment_range=range(1,4), 
                         subnetworks_extract='separate_index', selection_rate=1,
                         subnetworks_exrtact_basis=range(1,6),
                         save=False):
    # distance matrix
    _, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=True)
    dm = feature_engineering.normalize_matrix(dm)
    
    # parameters for construction of FM and RCM
    param = read_params(model, model_fm, model_rcm, folder=param)
    
    # global param_
    # param_ = param.copy()
    
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    labels = np.reshape(labels, -1)
    
    # features
    fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_exrtact_basis)
    alpha_global_averaged = fcs_global_averaged['alpha']
    beta_global_averaged = fcs_global_averaged['beta']
    gamma_global_averaged = fcs_global_averaged['gamma']
    
    alpha_global_averaged_rebuilded = cm_rebuild(alpha_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
    beta_global_averaged_rebuilded = cm_rebuild(beta_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
    gamma_global_averaged_rebuilded = cm_rebuild(gamma_global_averaged, dm, param, model, model_fm, model_rcm, True, False)
    
    strength_alpha = np.sum(np.abs(alpha_global_averaged_rebuilded), axis=1)
    strength_beta = np.sum(np.abs(beta_global_averaged_rebuilded), axis=1)
    strength_gamma = np.sum(np.abs(gamma_global_averaged_rebuilded), axis=1)
    
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
            
            # Selected CM
            alpha_selected = alpha_rebuilded[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta_rebuilded[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma_rebuilded[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            global x_selected
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            x_selected = x_selected.reshape(x_selected.shape[0], -1)
            
            # svm evaluation
            svm_results = k_fold_cross_validation_ml(x_selected, labels, k_folds=5, model_type='svm')
            
            all_results_list.append(svm_results)
    
    print('Evaluation compelete\n')
    
    # calculate average
    result_keys = all_results_list[0].keys()
    avg_results = {key: np.mean([res[key] for res in all_results_list]) for key in result_keys}
    print(f'Average SVM Results: {avg_results}')
    
    # save to xlsx
    identifier = f'{model}_{model_fm}_{model_rcm}'
    save_to_xlsx(all_results_list, subject_range, experiment_range,
                         folder_name='results_svm_channel_selection_evaluation',
                         file_name=f'svm_validation_{feature_cm}_by_{identifier}.xlsx',
                         sheet_name=f'{model}_sr_{selection_rate}')
            
    return all_results_list, avg_results

def svm_eval_circle_cw_fitting_intergrated(model_fm, model_rcm, selection_rate, feature, save=False):
    model = list(['exponential', 'gaussian', 'inverse', 'power_law', 'rational_quadratic', 'generalized_gaussian', 'sigmoid'])
    model = list(['gaussian', 'inverse', 'power_law', 'rational_quadratic', 'generalized_gaussian', 'sigmoid'])
    
    avgs_results_fitting = []
    for trail in range(0, 6):
        results_fitting, avg_results_fitting = svm_eval_circle_dmrc(projection_params={"source": "auto", "type": "3d_euclidean"}, 
                                                                    feature_cm=feature, 
                                                                    model=model[trail], model_fm=model_fm, model_rcm=model_rcm,
                                                                    param='fitted_results(sub1_sub5_joint_band)', 
                                                                    subject_range=range(6,16), experiment_range=range(1,4), 
                                                                    subnetworks_extract='separate_index', selection_rate=rate,
                                                                    subnetworks_exrtact_basis=range(1,6),
                                                                    save=save)
        
        avg_results_fitting = np.array([model[trail], avg_results_fitting['accuracy']])
        avgs_results_fitting.append(avg_results_fitting)
        
    avgs_results_fitting = np.vstack(avgs_results_fitting)
    avgs_results_fitting_df = pd.DataFrame(avgs_results_fitting)
    
    return avgs_results_fitting, avgs_results_fitting_df

# %% save
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

def save_to_xlsx(results, subject_range, experiment_range, folder_name, file_name, sheet_name):
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

# %%
if __name__ == '__main__':
    # selection_rate_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    selection_rate_list = [0.2]
    # for rate in selection_rate_list:
    #     all_results_list, avg_results = svm_eval_circle_original_fn(selection_rate=rate, feature_cm='pcc', 
    #                                                                 subject_range=range(6,16), experiment_range=range(1,4), 
    #                                                                 subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
    #                                                                 save=True)
    
    # for rate in selection_rate_list:
    #     all_results_list, avg_results = svm_eval_circle_GSLF(projection_params={"source": "auto", "type": "3d_euclidean"}, 
    #                                                          feature_cm='pcc', 
    #                                                          model='Generalized_Surface_Laplacian', model_config='basic', 
    #                                                          param='fitted_results_competing(sub1_sub5_joint_band)', kernel_normalization=False,
    #                                                          subject_range=range(6,16), experiment_range=range(1,4), selection_rate=rate,
    #                                                          subnetworks_extract='read', subnetworks_exrtact_basis=range(1,6),
    #                                                          save=True)

    for rate in selection_rate_list:
        svm_eval_circle_cw_fitting_intergrated(model_fm='advanced', model_rcm='linear_ratio', 
                                               selection_rate=rate, feature='pcc', save=True)
