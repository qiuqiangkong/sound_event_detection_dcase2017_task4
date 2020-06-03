import os
import sys
import numpy as np
import argparse
import h5py
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib as mpl

from utilities import (create_folder, get_filename, create_logging)
import config

 
def plot_clipwise_at_sed(args):
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    holdout_fold = '1'
    
    iterations = np.arange(0, 30001, 100)
    metric_types = ['at_map', 'sed_map', 'er']
    locs = [4, 1, 1]
    
    # Paths
    save_out_path = 'results/clipwise_at_sed_{}.pdf'.format(data_type)
    create_folder(os.path.dirname(save_out_path))
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        
    def _load_metrics(filename, model_type, loss_type, augmentation, batch_size, 
        data_type, metric_type):

        statistics_path = os.path.join(workspace, 'statistics', 
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            'statistics.pkl')
        
        statistics_dict = pickle.load(open(statistics_path, 'rb'))
        if metric_type == 'at_map':
            average_precision_matrix = np.array([statistics['clipwise_ap'] 
                for statistics in statistics_dict[data_type]])    # (N, classes_num)

            metrics = np.mean(average_precision_matrix, axis=-1)
            
        elif metric_type == 'sed_map':
            average_precision_matrix = np.array([statistics['framewise_ap'] 
                for statistics in statistics_dict[data_type]])    # (N, classes_num)
            
            metrics = np.mean(average_precision_matrix, axis=-1)
            
        elif metric_type == 'er':
            metrics = np.array([statistics['sed_metrics']['overall']['error_rate']['error_rate'] 
                for statistics in statistics_dict[data_type]])    # (N, classes_num)

            last_D = statistics_dict[data_type][-1]['sed_metrics']['overall']['error_rate']['deletion_rate']
            last_I = statistics_dict[data_type][-1]['sed_metrics']['overall']['error_rate']['insertion_rate']
            last_S = statistics_dict[data_type][-1]['sed_metrics']['overall']['error_rate']['substitution_rate']
            
        return metrics
        
    for m in range(3):
        lines = []
        linewidth = 0.8
        metric_type = metric_types[m]
   
        metrics = _load_metrics('main', 'Cnn_9layers_FrameMax', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='r', label='Cnn_9layers_FrameMax')
        lines.append(line)

        metrics = _load_metrics('main', 'Cnn_9layers_FrameAvg', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='b', label='Cnn_9layers_FrameAvg')
        lines.append(line)

        metrics = _load_metrics('main', 'Cnn_9layers_FrameAtt', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='g', label='Cnn_9layers_FrameAtt')
        lines.append(line)

        metrics = _load_metrics('main', 'Cnn_9layers_Gru_FrameAvg', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='k', label='Cnn_9layers_Gru_FrameAvg')
        lines.append(line)

        metrics = _load_metrics('main', 'Cnn_9layers_Gru_FrameAtt', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='m', label='Cnn_9layers_Gru_FrameAtt')
        lines.append(line)

        metrics = _load_metrics('main', 'Cnn_9layers_Transformer_FrameAvg', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='c', label='Cnn_9layers_Transformer_FrameAvg')
        lines.append(line)

        metrics = _load_metrics('main', 'Cnn_9layers_Transformer_FrameAtt', 'clip_bce', 'mixup', 32, data_type, metric_type)
        metrics = metrics[0 : 61]
        line, = axs[m].plot(metrics, linewidth=linewidth, c='y', label='Cnn_9layers_Transformer_FrameAtt')
        lines.append(line)
        
        if metric_type in ['at_map', 'sed_map']:
            axs[m].set_ylim(0, 0.9)
            axs[m].yaxis.set_ticks(np.arange(0., 0.91, 0.05))
            axs[m].yaxis.set_ticklabels(['0', '', '0.10', '', '0.20', '', '0.30', '', '0.40', '', '0.50', '', '0.60', '', '0.70', '', '0.80', '', '0.90'])

        elif metric_type == 'er':
            axs[m].set_ylim(0.5, 1.5)
            axs[m].yaxis.set_ticks(np.arange(0.50, 1.51, 0.05))
            axs[m].yaxis.set_ticklabels(['0.50', '', '0.60', '', '0.70', '', '0.80', '', '0.90', '', '1.00', '', '1.10', '', '1.20', '', '1.30', '', '1.40', '', '1.50'])
          
        iterations_per_save = 500
        axs[m].xaxis.set_ticks(np.arange(0, 61, 20))
        axs[m].xaxis.set_ticklabels([0, 20000, 40000, 60000])
        axs[m].set_xlim(0, len(metrics) - 1)
        axs[m].set_xlabel('iterations')
        axs[m].grid(color='k', linestyle='--', linewidth=0.3)
        
    axs[0].legend(handles=lines, loc=locs[0], fontsize=8)
    axs[1].legend(handles=lines, loc=locs[1], fontsize=8)
    axs[2].legend(handles=lines, loc=locs[2], fontsize=8)

    axs[0].set_title('AT mAP (macro)')
    axs[1].set_title('SED mAP (macro)')
    axs[2].set_title('SED ER (micro)')

    axs[0].set_ylabel('mAP')
    axs[1].set_ylabel('mAP')
    axs[2].set_ylabel('ER')

    plt.tight_layout()
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def plot_best_model_17_classes(args):
    
    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    
    holdout_fold = '1'
    samples_num = config.samples_num
    labels = config.labels
    iterations = np.arange(0, 30001, 100)
    metric_types = ['at_map', 'sed_map', 'er']
    locs = [4, 1, 1]
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '--', '--', '--',
        '--', '--', '--', '--', '--', ]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'navy', 'blueviolet', 'pink', 
        'lime', 'grey', 'orange', 'gold', 'peru', 'darkolivegreen', 'tan']
    
    # Paths
    save_out_path = 'results/best_model_17_classes_{}.pdf'.format(data_type)
    create_folder(os.path.dirname(save_out_path))
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    def _load_metrics(filename, model_type, loss_type, augmentation, 
            metric_type, batch_size, data_type):
        
        statistics_path = os.path.join(workspace, 'statistics', 
            '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
            'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size),
            'statistics.pkl')
            
        statistics_dict = pickle.load(open(statistics_path, 'rb'))

        if metric_type == 'at_map':
            average_precision_matrix = np.array([statistics['clipwise_ap'] 
                for statistics in statistics_dict[data_type]])    # (N, classes_num)

            metrics = average_precision_matrix

        elif metric_type == 'sed_map':
            average_precision_matrix = np.array([statistics['framewise_ap'] 
                for statistics in statistics_dict[data_type]])    # (N, classes_num)
            
            metrics = average_precision_matrix
        elif metric_type == 'er':
            er_mat = []
            for statistics in statistics_dict[data_type]:
                er = [statistics['sed_metrics']['class_wise'][label]['error_rate']['error_rate'] 
                    for label in labels]
                er_mat.append(er)

            metrics = np.array(er_mat)

        return metrics
        
    for m in range(3):
        metric_type = metric_types[m]

        metrics = _load_metrics('main', 'Cnn_9layers_Gru_FrameAtt', 'clip_bce', 'mixup', metric_type, 32, data_type)
        metrics = metrics[0 : 61, :]
        

        lines = []
        for k in range(metrics.shape[1]):
            line, = axs[m].plot(metrics[:, k], c=colors[k], linestyle=linestyles[k], linewidth=1.0, label='{}'.format(labels[k]))
            lines.append(line)

        if metric_type in ['at_map', 'sed_map']:
            axs[m].set_ylim(0, 1)
            axs[m].yaxis.set_ticks(np.arange(0., 1.01, 0.05))
            axs[m].yaxis.set_ticklabels(['0', '', '0.10', '', '0.20', '', 
                '0.30', '', '0.40', '', '0.50', '', '0.60', '', '0.70', '', 
                '0.80', '', '0.90', '', '1.00'])
        elif metric_type == 'er':
            axs[m].set_ylim(0., 3.5)
            axs[m].yaxis.set_ticks(np.arange(0.0, 3.51, 0.2))
            axs[m].yaxis.set_ticklabels(['{:.2f}'.format(e) for e in np.arange(0.0, 3.51, 0.2)])
        
        iterations_per_save = 100
        axs[m].xaxis.set_ticks(np.arange(0, 61, 20))
        axs[m].xaxis.set_ticklabels([0, 20000, 40000, 60000])
        axs[m].set_xlabel('iterations')
        axs[m].set_xlim(0, len(metrics) - 1)
        axs[m].grid(color='k', linestyle='--', linewidth=0.3)

    axs[0].set_title('Classwise AT mAP (macro)')
    axs[1].set_title('Classwise SED mAP (macro)')
    axs[2].set_title('Classwise SED ER (micro)')

    axs[0].set_ylabel('mAP')
    axs[1].set_ylabel('mAP')
    axs[2].set_ylabel('ER')

    box0 = axs[0].get_position()
    box1 = axs[1].get_position()
    box2 = axs[2].get_position()
    axs[2].legend(handles=lines, bbox_to_anchor=(1, 1), fontsize=8)

    plt.tight_layout()
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


def _get_threshold_index(thresholds):
    theta = 0
    indexes = []
    for j, threshold in enumerate(thresholds):
        if threshold >= theta:
            indexes.append(j)
            theta += 0.02
    return indexes


def plot_precision_recall_curve(args):

    workspace = args.workspace
    filename = 'main'
    holdout_fold = 1
    model_type = 'Cnn_9layers_Gru_FrameAtt'
    loss_type = 'clip_bce'
    augmentation = 'mixup'
    batch_size = 32
    iteration = 70000

    classes_num = config.classes_num
    labels = config.labels
 
    # Paths
    test_hdf5_path = os.path.join(workspace, 'hdf5s', 'testing.h5')
    evaluate_hdf5_path = os.path.join(workspace, 'hdf5s', 'evaluation.h5')

    predictions_dir = os.path.join(workspace, 'predictions', 
        '{}'.format(filename), 'holdout_fold={}'.format(holdout_fold), 
        'model_type={}'.format(model_type), 'loss_type={}'.format(loss_type), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    save_out_path = 'results/precision_recall_curve.pdf'
    create_folder(os.path.dirname(save_out_path))

    fig, axs = plt.subplots(3, 6, sharex=True, figsize=(12, 5.5))
    colors = ['b', 'r']
    for m, data_type in enumerate(['test', 'evaluate']):

        if data_type == 'test':
            hdf5_path = test_hdf5_path
        elif data_type == 'evaluate':
            hdf5_path = evaluate_hdf5_path

        with h5py.File(hdf5_path, 'r') as hf:
            target = hf['target'][:].astype(np.float32)

        prediction_path = os.path.join(predictions_dir, 
            '{}_iterations.prediction.{}.pkl'.format(iteration, data_type))

        # Load predictions
        output_dict = pickle.load(open(prediction_path, 'rb'))
        clipwise_output = output_dict['clipwise_output']

        for k in range(classes_num):
            (prec, recall, thresholds) = metrics.precision_recall_curve(target[:, k], clipwise_output[:, k])
            indexes = _get_threshold_index(thresholds)
            prec = prec[indexes]
            recall = recall[indexes]
            thresholds = thresholds[indexes]

            for i1 in range(len(thresholds)):
                axs[k // 6, k % 6].scatter(recall[i1], prec[i1], s=10, c=colors[m], marker='+', label='first', alpha=thresholds[i1])

            axs[k // 6, k % 6].plot(recall, prec, c='grey', linewidth=0.8, alpha=0.5)
            axs[k // 6, k % 6].set_title(labels[k], fontsize=8)
            axs[k // 6, k % 6].set_xlim(0, 1)
            axs[k // 6, k % 6].set_ylim(0, 1.01)
            axs[k // 6, k % 6].set_xlabel('Recall', fontsize=8)
            axs[k // 6, k % 6].set_ylabel('Precision', fontsize=8)

        axs[2, 5].set_visible(False)
        
        # Add color bar
        ax = fig.add_axes([0.88, 0.095, 0.008, 0.23])
        cmap = mpl.cm.Reds
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

        # Add color bar
        ax = fig.add_axes([0.94, 0.095, 0.008, 0.23])
        cmap = mpl.cm.Blues
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')

    plt.tight_layout()
    plt.savefig(save_out_path)
    print('Save figure to {}'.format(save_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_clipwise_plot = subparsers.add_parser('plot_clipwise_at_sed')
    parser_clipwise_plot.add_argument('--workspace', type=str, required=True)
    parser_clipwise_plot.add_argument('--data_type', type=str, required=True)

    parser_classwise_plot = subparsers.add_parser('plot_best_model_17_classes')
    parser_classwise_plot.add_argument('--workspace', type=str, required=True)
    parser_classwise_plot.add_argument('--data_type', type=str, required=True)

    parser_precision_recall_plot = subparsers.add_parser('plot_precision_recall_curve')
    parser_precision_recall_plot.add_argument('--workspace', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'plot_clipwise_at_sed':
        plot_clipwise_at_sed(args)

    elif args.mode == 'plot_best_model_17_classes':
        plot_best_model_17_classes(args)

    elif args.mode == 'plot_precision_recall_curve':
        plot_precision_recall_curve(args)
 
    else:
        raise Exception('Error argument!')