import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects

raw_csv_data = """model,provider,metric,seq_length,avg_tokens_per_second_mean,avg_tokens_per_second_std,total_peak_allocated_memory_MB_mean,total_peak_allocated_memory_MB_std
Stage1_num_head_3,huggingface,Throughput (tokens/sec),1024,47711.155,26.99885985,23670.96,0
Stage1_num_head_3,huggingface,Throughput (tokens/sec),2048,48434.5,58.16963641,39927.09,0
Stage1_num_head_3,huggingface,Throughput (tokens/sec),4096,0.0,0.0,0.0,0
Stage1_num_head_3,liger,Throughput (tokens/sec),1024,41195.8175,87.08984647,17628.62,0
Stage1_num_head_3,liger,Throughput (tokens/sec),2048,50201.19,23.78519007,17628.69,0
Stage1_num_head_3,liger,Throughput (tokens/sec),4096,50725.415,27.73761646,17628.82,0
Stage1_num_head_5,huggingface,Throughput (tokens/sec),1024,36343.9725,17.40839524,36127.93,0
Stage1_num_head_5,huggingface,Throughput (tokens/sec),2048,0.0,0.0,0.0,0
Stage1_num_head_5,huggingface,Throughput (tokens/sec),4096,0.0,0.0,0.0,0
Stage1_num_head_5,liger,Throughput (tokens/sec),1024,31624.2775,65.11460378,24091.17,0
Stage1_num_head_5,liger,Throughput (tokens/sec),2048,40167.1625,33.96462118,24091.24,0
Stage1_num_head_5,liger,Throughput (tokens/sec),4096,42939.525,5.467994757,24091.37,0
Stage2_num_head_3,huggingface,Throughput (tokens/sec),1024,21016.5625,13.20382615,26865.06,0
Stage2_num_head_3,huggingface,Throughput (tokens/sec),2048,20171.865,3.643089348,46333.24,0
Stage2_num_head_3,huggingface,Throughput (tokens/sec),4096,0.0,0.0,0.0,0
Stage2_num_head_3,liger,Throughput (tokens/sec),1024,20812.525,15.44039831,19684.75,0
Stage2_num_head_3,liger,Throughput (tokens/sec),2048,21743.91,3.564650521,21732.81,0
Stage2_num_head_3,liger,Throughput (tokens/sec),4096,19310.6275,1.550362861,27491.06,0
Stage2_num_head_5,huggingface,Throughput (tokens/sec),1024,18415.5325,3.228295474,39322.03,0
Stage2_num_head_5,huggingface,Throughput (tokens/sec),2048,12693.56,259.7861052,68938.27,0
Stage2_num_head_5,huggingface,Throughput (tokens/sec),4096,0.0,0.0,0.0,0
Stage2_num_head_5,liger,Throughput (tokens/sec),1024,18072.3775,37.29633617,26147.29,0
Stage2_num_head_5,liger,Throughput (tokens/sec),2048,19605.35,14.82546683,28195.36,0
Stage2_num_head_5,liger,Throughput (tokens/sec),4096,18052.26,2.558945095,32291.49,0
"""

csv_data = StringIO(raw_csv_data)

df = pd.read_csv(csv_data)
df.drop(columns=['metric'], inplace=True)
# Data cleanup

# df = df[df['avg_tokens_per_second_mean'] != 'OOM']
df['avg_tokens_per_second_mean'] = pd.to_numeric(df['avg_tokens_per_second_mean'], errors='coerce')
df['avg_tokens_per_second_std'] = pd.to_numeric(df['avg_tokens_per_second_std'], errors='coerce')
# df = df[df['total_peak_allocated_memory_MB_mean'] != 'OOM']
df['total_peak_allocated_memory_MB_mean'] = pd.to_numeric(df['total_peak_allocated_memory_MB_mean'], errors='coerce')
df['total_peak_allocated_memory_MB_std'] = pd.to_numeric(df['total_peak_allocated_memory_MB_std'], errors='coerce')

df.replace('OOM', np.nan, inplace=True)

# Replace NaN values with zero
df.fillna(0, inplace=True)

print(df.info())

models = df['model'].unique()
metrics = ['total_peak_allocated_memory_MB_mean', 'avg_tokens_per_second_mean']
std_metrics = ['total_peak_allocated_memory_MB_std', 'avg_tokens_per_second_std']

metric_to_display_label = {
    'total_peak_allocated_memory_MB_mean': {
        'y_label': "MB",
        'title': 'Peak Allocated Memory (MB)',
        'filename': 'Peak_Allocated_Memory',
    },
    'avg_tokens_per_second_mean': {
        "y_label": "tokens/sec",
        "title": "Throughput (tokens/sec)",
        'filename': 'Throughput',
    }
}

sns.set(style="whitegrid")

# Plot each metric for each model with error bars
for model_idx, model in enumerate(models):
    for metric_idx, (metric, std_metric) in enumerate(zip(metrics, std_metrics)):
        print(f"Model: {model}")
        plt.figure(figsize=(8, 6))
        
        data = df[df['model'] == model]
        ##################### Changes made to plot stderr    
        data = data.sort_values(['provider', 'seq_length'])
        ##################### Changes made to plot stderr    

        seq_length = data['seq_length']
        
        ax = sns.barplot(
            x='seq_length',
            y=metric,
            hue='provider',
            data=data,
            ci=None,
            palette=['#EA4335', '#4285F4'],
            hue_order=['huggingface', 'liger'],
        )
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

##################### Changes made to plot stderr        ##########################################
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        ax.errorbar(x=x_coords, y=y_coords, yerr=data[std_metric], fmt="none", c="k", capsize=5)
        
##########################################################################################################
        
        # Add labels to the bars
        # for p in ax.patches:
        #     if p.get_height() > 0 and p.get_width() > 0:
        #         ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width(),
        #                                           p.get_height()), ha='right', va='bottom',
        #                 fontsize=12, color='black')
        
        # Add the out-of-memory label
        y_min, y_max = ax.get_ylim()
        for p in ax.patches:
            height = p.get_height()

            # Add a label for empty bars to indicate OOM
            if height == 0:
                ax.annotate('OOM',  # Label to add
                            (p.get_x() + p.get_width() / 2, 0.05*y_max),  # x position: center of bar, y position: bar height
                            ha='center', va='bottom',  # Align horizontally and vertically
                            fontsize=16, color='red')
        
        
        max_height = max([p.get_height() for p in ax.patches])
        plt.ylim(0, max_height * 1.25)

        # plt.title(metric_to_display_label[metric]["title"], fontsize=18)
        plt.xlabel('Seq Length', fontsize=16)
        plt.xticks(fontsize=16)
        plt.ylabel(metric_to_display_label[metric]["y_label"], fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(title='', labels=['Hugging Face', 'Liger Kernel'], fontsize=16, loc='upper left')

        fig = plt.gcf()  # Get the current figure

        # fig.patch.set_edgecolor('black')  # Set the edge color (outline)
        # fig.patch.set_linewidth(2)
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        # plt.clf()
        
        plt.savefig(f"plots/{metric_to_display_label[metric]['filename']}{model}")