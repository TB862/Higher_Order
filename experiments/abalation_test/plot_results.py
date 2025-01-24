import os 
import sys 

directory_path = os.path.abspath(os.getcwd())

if directory_path not in sys.path:
    sys.path.append(directory_path)

import matplotlib.pyplot as plt 
import numpy as np 
import json 
import argparse
from itertools import product 
from typing import *   

PATH_TO_PARENT = os.path.join(os.getcwd(), 'experiments', 'abalation_test') 
DEFAULT_PATH = os.path.join(PATH_TO_PARENT, 'checkpoints') 


def plot_confusion_matrix(results: Dict, dataset: str, width, height, **ex_args):
    def plot(matrix, path, xlabel: str, ylabel: str):
        fig, ax = plt.subplots()

        cax = ax.matshow(matrix, cmap=plt.cm.Blues, vmin=np.min(matrix), vmax=np.max(matrix))
        cbar = fig.colorbar(cax)

        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        cbar.ax.tick_params(labelsize=12)

        N = len(matrix)
        labels = np.arange(N)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels + 1, fontsize=12)
        ax.set_yticks(labels)
        ax.set_yticklabels(labels + 1, fontsize=12)

        # Force x-labels to the bottom axis
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
        ax.set_title(dataset, fontsize=20, fontweight='bold')

        plt.xticks(rotation=45)
        plt.savefig(path)
        plt.close()

    if height is None:
        width = height 

    confusion_matrix = np.zeros((width, height))
    for setting in results:
        mean, std = results[setting][1]
        nhead1, nhead2 = setting
        confusion_matrix[nhead1-1, nhead2-1] = mean 

    save_path = os.path.join(PATH_TO_PARENT, f'confusion_matrix_{dataset}.pdf')
    plot(confusion_matrix, save_path, 'Heads in First Layer', 'Heads in Second layer')


def make_line_plot(results: Dict, dataset: str):
    fig = plt.figure(tight_layout=False, figsize=(8, 6))
    ax = fig.add_subplot()

    all_min_vals = []
    all_max_vals = []

    model_names = ['GRAND', 'HoGA-GRAND']

    for midx, model_result in enumerate(results):
        mean_vals = [] 
        std_devs = []  # List to store standard deviations for confidence intervals
        for nparam in model_result:
            config, (mean, std) = model_result[nparam]
            mean_vals.append(mean)
            std_devs.append(std)

        max_y = np.max(mean_vals)
        y_plots = np.array(mean_vals) / max_y
        y_errors = np.array(std_devs) / max_y  # Normalize standard deviations for confidence intervals

        min_plot = np.min(y_plots)
        max_plot = np.max(y_plots)
        ax.set_ylim([min_plot, max_plot])

        all_min_vals.append(min_plot)
        all_max_vals.append(max_plot)

        xticks = list(model_result.keys())
        ax.plot(xticks, y_plots, marker='o', linewidth=3, label=model_names[midx])

        # Plot confidence intervals as shaded regions
        ax.fill_between(xticks, y_plots - y_errors, y_plots + y_errors, alpha=0.3)

    # Set y-ticks with two ticks and adjust the size to match the bar plot
    min_plot = min(all_min_vals)
    max_plot = max(all_max_vals) 
    yticks = [min_plot, max_plot]
    ax.set_yticks(yticks)
    eps = 0.005
    ax.set_ylim([min_plot-eps, 1.0+eps])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # Set x-ticks with two ticks and adjust the size to match the bar plot
    xticks = [int(xticks[0]), int(xticks[-1])]
    eps_x = 0.005
    ax.set_xlim([xticks[0]-eps_x, xticks[1]+eps_x])

    if dataset == 'Cora' or True:
        ax.set_xticks(xticks)    

    # Set tick parameters to match the size from the first bar plot
    plt.tick_params(axis='both', which='major', labelsize=40)

    # Set labels and title
    #plt.ylabel('Relative Accuracy', fontsize=40)
    plt.title(dataset, fontsize=40, weight='bold')
    plt.ylabel('Relative Accuracy', fontsize=40)
    if dataset == 'Cora' or True:
        plt.xlabel('Number of layers', fontsize=40)
    
    if dataset == 'Cora':
        plt.legend(fontsize=25)  # Adjust legend as needed

    save_path = os.path.join(PATH_TO_PARENT, f'line_plot_{dataset}.pdf')
    plt.savefig(save_path)
    plt.close()



def make_line_plot_old(results: Dict, dataset: str):
    fig = plt.figure(tight_layout=False, figsize=(8, 6))
    ax = fig.add_subplot() 
    
    bar_width = 0.35  # width of the bars

    all_mean_vals = []
    all_std_devs = []  # List to store all standard deviations

    for midx, model_result in enumerate(results):
        mean_vals = [] 
        std_devs = []  # List to store the standard deviations for this model
        for nparam in model_result:
            config, (mean, std) = model_result[nparam]
            mean_vals.append(100 * mean)
            std_devs.append(100 * std)  # Store the standard deviation

        all_mean_vals.extend(mean_vals)  # Collecting all mean values
        all_std_devs.extend(std_devs)  # Collecting all standard deviations

        xticks = list(range(len(model_result.keys())))  # positions for the bars
        ax.bar(xticks, mean_vals, bar_width, label=f'Model {midx}', color='purple')

        # Add error bars for standard deviation
        ax.errorbar(xticks, mean_vals, yerr=std_devs, fmt='none', ecolor='black', capsize=5)

    # Set y-axis limits to the overall min and max values plus/minus the maximum standard error
    max_std_dev = np.max(all_std_devs)
    min_val = np.min(all_mean_vals) - max_std_dev - 0.1
    max_val = np.max(all_mean_vals) + max_std_dev - 0.01
    ax.set_ylim(min_val, max_val)

    ytick_interval = (max_val - min_val) / 2  # You can adjust the divisor to increase/decrease the number of ticks
    yticks = np.arange(min_val, max_val + ytick_interval/2, ytick_interval)
    ax.set_yticks(yticks)

    # Set labels and title
    #plt.ylabel('Accuracy', fontsize=40)
    #plt.title(dataset, fontsize=40, weight='bold')
    plt.tick_params(axis='both', which='major', labelsize=40)

    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    xticks = ax.get_xticks()
    plt.ylabel('Accuracy', fontsize=40)
    if dataset == 'Pubmed' or True:
        plt.xlabel('K hops', fontsize=40)
        ax.set_xticks([0, int(xticks[-1])-1])
        ax.set_xticklabels([1, int(xticks[-1])])
    else:
        ax.set_xticks([])
    
    plt.title(dataset, weight='bold', fontsize=40)

    if dataset == 'Cora':
        plt.legend(['HoGA-GRAND'], fontsize=20)
    
    save_path = os.path.join(PATH_TO_PARENT, f'bar_plot_{dataset}.pdf')
    plt.savefig(save_path)
    plt.close()


def load_in(nparam: List, path_to: str, model_name: str = '', expr_type='khop') -> Dict[List, Any]:
    param_nm = path_to.split('_')[-1]
    out_dict = {} 
    for setting in nparam:
        #expr_type = 'layer'
        if expr_type == 'khop' and False:
            file_name = os.path.join(path_to, f'{model_name}_K_hops_{setting}')
        else: 
            file_name = os.path.join(path_to, f'{model_name}_num_layers_{setting}_num_heads_{[8 if i == 0 else 1 for i in range(setting)]}')      # hard coding 

        # load in performance 
        with open(file_name+'.txt', 'r') as acc_reader:
            ms_line = acc_reader.readline().split(',')
            mean, std = float(ms_line[0].split(':')[1]), float(ms_line[1].split(':')[1])

        # now json 
        with open(file_name+'.json', 'r') as json_reader:
            config = json.load(json_reader)

        out_dict[setting] = [config, (mean, std)]

    return out_dict


def collect_config_dict(model_names: List[str], dataset: str, lbl: str, path: str, nparam: int):
    configs = [] 
    for mn in model_names:
        folder_nm = f'{dataset}_{lbl}'
        full_path = os.path.join(path, folder_nm)

        temp_dict = load_in(nparam, full_path, mn)
        configs.append(temp_dict)

    return configs 

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--dataset', type=str, help='Dataset to plot')
    args.add_argument('--nheads', type=int, help='Number of heads', default=1)
    args.add_argument('--nlayers', type=int, help='Number of layers', default=1)
    args.add_argument('--khops', type=int, help='Number of layers', default=1)
    args.add_argument('--plot_line', help='Dataset to plot', default=False, action='store_true')
    args.add_argument('--plot_bar', help='Dataset to plot', default=False, action='store_true')
    args.add_argument('--path', type=str, help='Path to the experiment results', default=DEFAULT_PATH)
    args.add_argument('--plot_confusion', help='If added, plot the confusion matrix', action='store_true', default=False)
    args.add_argument('--parameter', type=str, help='If added, plot the optimal parameter', default=None)
    pargs = args.parse_args()

    nheads = np.array([np.array(setting)+1 for setting in product(range(pargs.nheads), range(pargs.nheads))])
    nlayers = np.array([l+1 for l in range(pargs.nlayers)]) 
    khops = np.array([k+1 for k in range(pargs.khops)])
    dataset = pargs.dataset 
    path = pargs.path

    if pargs.khops > 1:
        model_names = ['MultiHop_GAT']
        expr_type = 'khop'
        nparam = khops 
    elif pargs.nlayers > 1:
        model_names = ['GAT', 'MultiHop_GAT']
        expr_type = 'layers'
        nparam = nlayers 
    
    plot_confusion = pargs.plot_confusion 
    plot_line = pargs.plot_line 
    plot_bar = pargs.plot_bar 
    
    # arg plot here 
    pname = pargs.parameter 
    plot_param = (pname is None)
    
    folder_nm = dataset 
    if np.sum([plot_line, plot_bar]) > 0:
        
        config_dict = collect_config_dict(model_names, dataset, expr_type, path, nparam)
        
        make_line_plot(config_dict, dataset)


    if plot_confusion:
        model_names = ['MultiHop_GAT']
        head_config_dict = collect_config_dict(model_names, dataset, 'heads', path, nheads)   # broken bec list 
        width, height = max(nheads[:, 0]), max(nheads[:, 1])
        plot_confusion_matrix(head_config_dict, dataset, width, height)  

    
