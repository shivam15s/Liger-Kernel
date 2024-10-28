import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects

raw_csv_data = """model,provider,batch_size,total_peak_allocated_memory_MB_mean,total_peak_allocated_memory_MB_std,ms_time_mean,ms_time_std
ORPO,huggingface,2.0,5.414666,0.0,22.396347,0.097983
ORPO,liger,2.0,4.567564,0.0,21.667089,0.041439
ORPO,huggingface,4.0,10.161512,0.0,44.011051,0.130565
ORPO,liger,4.0,4.592796,0.0,42.959969,0.05401
ORPO,huggingface,8.0,19.655598,0.0,87.401718,0.264362
ORPO,liger,8.0,4.643258,0.0,85.299039,0.047381
ORPO,huggingface,16.0,38.643507,0.0,176.205963,0.249156
ORPO,liger,16.0,4.744184,0.0,170.257757,0.378791
ORPO,huggingface,32.0,76.619194,0.0,352.675191,0.857656
ORPO,liger,32.0,4.94628,0.0,341.971476,0.224487
ORPO,huggingface,64.0,OOM,0.0,OOM,0.0
ORPO,liger,64.0,5.349605,0.0,683.753845,0.510219
ORPO,huggingface,128.0,OOM,0.0,OOM,0.0
ORPO,liger,128.0,6.158057,0.0,1368.296509,0.556836
"""

csv_data = StringIO(raw_csv_data)

df = pd.read_csv(csv_data)

# Data cleanup

# df = df[df['ms_time_mean'] != 'OOM']
df['ms_time_mean'] = pd.to_numeric(df['ms_time_mean'], errors='coerce')
df['ms_time_std'] = pd.to_numeric(df['ms_time_std'], errors='coerce')
# df = df[df['total_peak_allocated_memory_MB_mean'] != 'OOM']
df['total_peak_allocated_memory_MB_mean'] = pd.to_numeric(df['total_peak_allocated_memory_MB_mean'], errors='coerce')
df['total_peak_allocated_memory_MB_std'] = pd.to_numeric(df['total_peak_allocated_memory_MB_std'], errors='coerce')

df.replace('OOM', np.nan, inplace=True)

# Replace NaN values with zero
df.fillna(0, inplace=True)
df['total_peak_allocated_memory_MB_mean'] = df['total_peak_allocated_memory_MB_mean'] * 1000
df['total_peak_allocated_memory_MB_std'] = df['total_peak_allocated_memory_MB_std'] * 1000
df['batch_size'] = df['batch_size'].astype(int)
print(df.info())

models = df['model'].unique()
metrics = ['total_peak_allocated_memory_MB_mean', 'ms_time_mean']
std_metrics = ['total_peak_allocated_memory_MB_std', 'ms_time_std']

metric_to_display_label = {
    'total_peak_allocated_memory_MB_mean': {
        'y_label': "MB",
        'title': 'Peak Allocated Memory (MB)',
        'filename': 'mem',
    },
    'ms_time_mean': {
        "y_label": "ms",
        "title": "Time (ms)",
        'filename': 'tps',
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
        data = data.sort_values(['provider', 'batch_size'])
        ##################### Changes made to plot stderr    

        batch_size = data['batch_size']
        ax = sns.barplot(
            x='batch_size',
            y=metric,
            hue='provider',
            data=data,
            ci=None,
            palette=['#EA4335', '#4285F4'],
            hue_order=['huggingface', 'liger'],
        )
##################### Changes made to plot stderr        ##########################################
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        print(len(y_coords))
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
                            (p.get_x() + p.get_width() / 2.2, 0.05*y_max),  # x position: center of bar, y position: bar height
                            ha='center', va='bottom',  # Align horizontally and vertically
                            fontsize=12, color='red')
        
        
        max_height = max([p.get_height() for p in ax.patches])
        plt.ylim(0, max_height * 1.25)

        # plt.title(metric_to_display_label[metric]["title"], fontsize=18)
        plt.xlabel('Batch Size', fontsize=16)
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
        
        
        
        plt.savefig(f"{model}_{metric_to_display_label[metric]['filename']}")

