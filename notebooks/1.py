import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split(',')
        prune = float(parts[0].split(': ')[1])
        value = float(parts[1].split(': ')[1])
        data.append((prune, value))
    return pd.DataFrame(data, columns=['PrunePercent', filename.split('.')[0]])

# Load all files
bleu_files = ['class_distribution_bleu_scores.txt', 'class_uniform_bleu_scores.txt', 'global_bleu_scores.txt']
ppl_files = ['class_distribution_perplexity.txt', 'class_uniform_perplexity.txt', 'global_perplexity.txt']

bleu_dfs = [load_data(f) for f in bleu_files]
ppl_dfs = [load_data(f) for f in ppl_files]

# Merge data
bleu_df = bleu_dfs[0]
for df in bleu_dfs[1:]:
    bleu_df = bleu_df.merge(df, on='PrunePercent')

ppl_df = ppl_dfs[0]
for df in ppl_dfs[1:]:
    ppl_df = ppl_df.merge(df, on='PrunePercent')

# Clean column names
bleu_df.columns = ['PrunePercent', 'Distribution_BLEU', 'Uniform_BLEU', 'Global_BLEU']
ppl_df.columns = ['PrunePercent', 'Distribution_PPL', 'Uniform_PPL', 'Global_PPL']

# Plotting
plt.style.use('seaborn')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# BLEU Scores
ax1.plot(bleu_df['PrunePercent'], bleu_df['Distribution_BLEU'], 'o-', label='Class Distribution')
ax1.plot(bleu_df['PrunePercent'], bleu_df['Uniform_BLEU'], 's-', label='Class Uniform')
ax1.plot(bleu_df['PrunePercent'], bleu_df['Global_BLEU'], 'd-', label='Global')
ax1.set_title('BLEU Scores vs Pruning Percentage')
ax1.set_xlabel('Pruning Percentage')
ax1.set_ylabel('BLEU Score')
ax1.legend()
ax1.grid(True)

# Perplexity (log scale)
ax2.plot(ppl_df['PrunePercent'], ppl_df['Distribution_PPL'], 'o-', label='Class Distribution')
ax2.plot(ppl_df['PrunePercent'], ppl_df['Uniform_PPL'], 's-', label='Class Uniform')
ax2.plot(ppl_df['PrunePercent'], ppl_df['Global_PPL'], 'd-', label='Global')
ax2.set_yscale('log')
ax2.set_title('Perplexity vs Pruning Percentage (Log Scale)')
ax2.set_xlabel('Pruning Percentage')
ax2.set_ylabel('Perplexity (log scale)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('bleu_and_perplexity.png')  # Save the combined BLEU and Perplexity plot
plt.show()

# Additional analysis: Performance drop points
fig, ax = plt.subplots(figsize=(10, 6))
for col in bleu_df.columns[1:]:
    ax.plot(bleu_df['PrunePercent'], bleu_df[col]/bleu_df[col].max()*100, 'o-', label=col.split('_')[0])
ax.set_title('Relative BLEU Score Retention (%)')
ax.set_xlabel('Pruning Percentage')
ax.set_ylabel('Percentage of Maximum BLEU Score')
ax.legend()
ax.grid(True)
plt.savefig('relative_bleu_score_retention.png')  # Save the relative BLEU score retention plot
plt.show()

# Perplexity growth rate analysis
fig, ax = plt.subplots(figsize=(10, 6))
for col in ppl_df.columns[1:]:
    ax.plot(ppl_df['PrunePercent'], np.log10(ppl_df[col]), 'o-', label=col.split('_')[0])
ax.set_title('Log10 Perplexity Growth')
ax.set_xlabel('Pruning Percentage')
ax.set_ylabel('log10(Perplexity)')
ax.legend()
ax.grid(True)
plt.savefig('log_perplexity_growth.png')  # Save the log10 perplexity growth plot
plt.show()

# Combined performance-efficiency plot
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Pruning Percentage')
ax1.set_ylabel('BLEU Score', color=color)
for col in bleu_df.columns[1:]:
    ax1.plot(bleu_df['PrunePercent'], bleu_df[col], 'o-', color=color, label=f'{col.split("_")[0]} BLEU')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Perplexity (log scale)', color=color)
for col in ppl_df.columns[1:]:
    ax2.plot(ppl_df['PrunePercent'], ppl_df[col], 's--', color=color, label=f'{col.split("_")[0]} PPL')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.title('BLEU and Perplexity Trade-off')
plt.savefig('bleu_perplexity_tradeoff.png')  # Save the BLEU and Perplexity trade-off plot
plt.show()
