import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects

raw_csv_data = """model,provider,batch_size,total_peak_allocated_memory_MB_mean,total_peak_allocated_memory_MB_std,avg_tokens_per_second_mean,avg_tokens_per_second_std
phi3,liger,64,27675.61,0.1280624847,19034.044,347.5518359
phi3,liger,128,42812.514,0.3564828187,19541.188,345.1854951
phi3,liger,192,58093.44,0,20346.198,226.2334207
phi3,huggingface,64,30367.356,265.4749648,15838.902,268.5067193
phi3,huggingface,128,49431.36,0.1369306394,16663.028,191.4847415
phi3,huggingface,192,OOM,,OOM,
mistral_7b,liger,64,36866.466,0.2370232056,13473.004,421.4440648
mistral_7b,liger,128,50471.372,0.02683281573,14318.87,316.8447045
mistral_7b,liger,192,64076.36,0,14400.042,86.19178308
mistral_7b,huggingface,64,42876.68,0.402492236,11195.926,278.8630242
mistral_7b,huggingface,128,64101.93,0,11300.7325,232.5714327
mistral_7b,huggingface,192,OOM,,OOM,
llama,liger,32,22926.4,951.7,11410.6,223.6
llama,liger,48,27259.6,0.1,11542.9,207.2
llama,liger,64,30225.2,1547.3,12268.6,230.5
llama,huggingface,32,38544.7,2731.1,9601.6,347.6
llama,huggingface,48,56043.6,1922.4,9380,264.5
llama,huggingface,64,66895.2,7562.1,8588.7,275.4
qwen,liger,32,22641.8,527.4,11835.4,385.6
qwen,liger,48,25732.4,1163.2,12555.4,259.4
qwen,liger,64,28544.6,1957.7,12963.6,309.6
qwen,huggingface,32,42594.8,4923.8,9950.4,429.1
qwen,huggingface,48,59555.3,3837,10001,471.5
qwen,huggingface,64,OOM,,OOM,
gemma_7b,liger,32,27078,63.8,8686.4,258.7
gemma_7b,liger,48,30421.8,46.8,9139.5,304.4
gemma_7b,liger,64,34668.3,0.4,9365.4,126.3
gemma_7b,huggingface,32,56147.2,2794.7,7126.3,233
gemma_7b,huggingface,48,OOM,,OOM,
gemma_7b,huggingface,64,OOM,,OOM,"""

csv_data = StringIO(raw_csv_data)

df = pd.read_csv(csv_data)

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
        'filename': 'mem',
    },
    'avg_tokens_per_second_mean': {
        "y_label": "tokens/sec",
        "title": "Throughput (tokens/sec)",
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
        
        plt.savefig(f"plots/{model}_{metric_to_display_label[metric]['filename']}")