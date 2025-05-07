import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Load and parse sparsity data
def parse_sparsity_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    data = defaultdict(list)
    current_prune = None
    
    for line in content.split('\n'):
        if not line.strip():
            continue
        if line.startswith('PrunePercent:'):
            current_prune = float(line.split(': ')[1].split(',')[0])
            continue
        
        parts = line.split(', ')
        for part in parts:
            if 'Sparsity:' in part:
                layer = part.split(': ')[0].strip()
                sparsity = float(part.split(': ')[2].replace('%', '').strip())
                data[current_prune].append((layer, sparsity))
    print(data)
    return data

# Load all files
global_sparsity = parse_sparsity_file('global_sparsity.txt')
class_dist_sparsity = parse_sparsity_file('class_distribution_sparsity.txt')

# Create DataFrames for analysis
def create_sparsity_df(sparsity_data, method_name):
    rows = []
    for prune_pct, layers in sparsity_data.items():
        for layer, sparsity in layers:
            # Categorize layers
            if 'encoder' in layer:
                component = 'encoder'
            elif 'decoder' in layer:
                component = 'decoder'
            else:
                component = 'other'
            
            # Further breakdown
            if 'self_attn' in layer:
                subcomponent = 'self_attention'
            elif 'encoder_attn' in layer:
                subcomponent = 'encoder_attention'
            elif 'fc' in layer or 'dense' in layer:
                subcomponent = 'feed_forward'
            elif 'norm' in layer or 'layer_norm' in layer:
                subcomponent = 'normalization'
            elif 'embed' in layer:
                subcomponent = 'embedding'
            else:
                subcomponent = 'other'
            
            # Projection type
            if 'q_proj' in layer:
                proj_type = 'query'
            elif 'k_proj' in layer:
                proj_type = 'key'
            elif 'v_proj' in layer:
                proj_type = 'value'
            elif 'out_proj' in layer:
                proj_type = 'output'
            else:
                proj_type = None
            
            rows.append({
                'prune_percent': prune_pct,
                'layer': layer,
                'sparsity': sparsity,
                'method': method_name,
                'component': component,
                'subcomponent': subcomponent,
                'proj_type': proj_type,
                'layer_num': int(re.search(r'layers\.(\d+)', layer).group(1)) if re.search(r'layers\.(\d+)', layer) else -1
            })
    
    return pd.DataFrame(rows)

global_df = create_sparsity_df(global_sparsity, 'global')
print("Hello guys, i'm DIngu the mad girl")
print(global_df.head())  # Debugging line to check the DataFrame structure
class_dist_df = create_sparsity_df(class_dist_sparsity, 'class_distribution')
combined_df = pd.concat([global_df, class_dist_df])

# Plotting functions
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [12, 8]

# Plot for sparsity by component
def plot_sparsity_by_component(df, method_name):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Sparsity Distribution by Component ({method_name})', y=1.02)

    # Encoder and Decoder Components
    for ax, component in zip(axes[0], ['encoder', 'decoder']):
        comp_df = df[df['component'] == component]
        grouped = comp_df.groupby(['prune_percent', 'subcomponent'])['sparsity'].mean().unstack()
        grouped.plot(kind='bar', ax=ax)
        ax.set_title(f'{component.capitalize()} Components')
        ax.set_xlabel('Pruning Percentage')
        ax.set_ylabel('Average Sparsity (%)')
        ax.legend(title='Subcomponent')
        ax.grid(True)

    # Attention Projections
    for ax, proj_type in zip(axes[1], ['query', 'key', 'value', 'output']):
        proj_df = df[df['proj_type'] == proj_type]
        if proj_df.empty:
            ax.set_visible(False)
            continue
        grouped = proj_df.groupby(['prune_percent', 'component'])['sparsity'].mean().unstack()
        grouped.plot(kind='bar', ax=ax)
        ax.set_title(f'{proj_type.capitalize()} Projections')
        ax.set_xlabel('Pruning Percentage')
        ax.set_ylabel('Average Sparsity (%)')
        ax.legend(title='Component')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{method_name}_sparsity_by_component.png')  # Save the image as PNG
    plt.close()  # Close the plot to free memory

# Plot for each method
plot_sparsity_by_component(global_df, 'Global Pruning')
plot_sparsity_by_component(class_dist_df, 'Class Distribution Pruning')

# Compare methods for key components
def compare_pruning_methods():
    components = ['self_attention', 'encoder_attention', 'feed_forward', 'embedding']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sparsity Comparison Between Pruning Methods', y=1.02)

    for ax, component in zip(axes.flatten(), components):
        global_group = global_df[global_df['subcomponent'] == component].groupby('prune_percent')['sparsity'].mean()
        class_group = class_dist_df[class_dist_df['subcomponent'] == component].groupby('prune_percent')['sparsity'].mean()
        combined = pd.DataFrame({'Global': global_group, 'Class Distribution': class_group})
        combined.plot(kind='bar', ax=ax)
        ax.set_title(component.replace('_', ' ').title())
        ax.set_xlabel('Pruning Percentage')
        ax.set_ylabel('Average Sparsity (%)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('sparsity_comparison_between_methods.png')  # Save the image as PNG
    plt.close()  # Close the plot to free memory

compare_pruning_methods()

# Layer-wise sparsity heatmap
def plot_layerwise_heatmap(df, method_name, prune_percent):
    plt.figure(figsize=(16, 10))
    
    # Filter and pivot
    filtered = df[(df['prune_percent'] == prune_percent) & 
                 (df['subcomponent'].isin(['self_attention', 'encoder_attention', 'feed_forward']))]
    
    # Create meaningful layer labels
    filtered['layer_label'] = filtered.apply(
        lambda x: f"{x['component'][:3]}.L{x['layer_num']}.{x['proj_type'] if x['proj_type'] else x['subcomponent'][:4]}", 
        axis=1)
    
    pivot = filtered.pivot_table(values='sparsity', index='layer_label', columns='subcomponent')
    
    # Plot heatmap
    plt.imshow(pivot, cmap='viridis', aspect='auto')
    plt.colorbar(label='Sparsity (%)')
    plt.title(f'Layer-wise Sparsity at {prune_percent}% Pruning ({method_name})')
    plt.yticks(range(len(pivot)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    
    # Annotate values
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.1f}", 
                     ha='center', va='center', color='w')
    
    plt.tight_layout()
    plt.savefig(f'layerwise_sparsity_{method_name}_{prune_percent}.png')  # Save the image as PNG
    plt.close()  # Close the plot to free memory

# Example heatmaps
for prune_pct in [30, 50, 70]:
    plot_layerwise_heatmap(global_df, 'Global Pruning', prune_pct)
    plot_layerwise_heatmap(class_dist_df, 'Class Distribution Pruning', prune_pct)

# Sparsity variance analysis
def plot_sparsity_variance():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for method, df in [('Global', global_df), ('Class Distribution', class_dist_df)]:
        variance = df.groupby(['prune_percent', 'layer'])['sparsity'].mean().groupby('prune_percent').std()
        ax.plot(variance.index, variance.values, 'o-', label=method)
    
    ax.set_title('Standard Deviation of Sparsity Across Layers')
    ax.set_xlabel('Pruning Percentage')
    ax.set_ylabel('Std Dev of Sparsity (%)')
    ax.legend()
    ax.grid(True)
    plt.savefig('sparsity_variance.png')  # Save the image as PNG
    plt.close()  # Close the plot to free memory

plot_sparsity_variance()

# Sparsity vs. BLEU correlation (using previously loaded BLEU scores)
def plot_sparsity_bleu_correlation():
    # Load BLEU scores (from previous analysis)
    bleu_data = {
        'global': {10: 27.50, 20: 27.11, 30: 26.39, 40: 22.61, 50: 6.49, 60: 2.03, 70: 1.08},
        'class_distribution': {10: 27.66, 20: 27.07, 30: 25.86, 40: 22.90, 50: 10.86, 60: 7.86, 70: 2.26}
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method, bleu_scores in bleu_data.items():
        if method == 'global':
            df = global_df
        else:
            df = class_dist_df
        
        # Calculate average sparsity for each prune percentage
        avg_sparsity = df.groupby('prune_percent')['sparsity'].mean()
        
        # Get corresponding BLEU scores
        prune_levels = sorted(bleu_scores.keys())
        sparsity = [avg_sparsity[p] for p in prune_levels]
        bleu = [bleu_scores[p] for p in prune_levels]
        
        ax.plot(sparsity, bleu, 'o-', label=method.replace('_', ' ').title())
    
    ax.set_title('Model Performance vs. Average Sparsity')
    ax.set_xlabel('Average Sparsity (%)')
    ax.set_ylabel('BLEU Score')
    ax.legend()
    ax.grid(True)
    plt.savefig('sparsity_bleu_correlation.png')  # Save the image as PNG
    plt.close()  # Close the plot to free memory

plot_sparsity_bleu_correlation()
