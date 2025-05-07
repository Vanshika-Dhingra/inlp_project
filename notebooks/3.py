import matplotlib.pyplot as plt

# Manually extracted from the file
prune_percents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bleu_scores = [29.91, 29.5, 27.12, 26.04, 15.81, 5.97, 2.12, 1.07, 0.0, 0.0]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(prune_percents, bleu_scores, color='mediumslateblue', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('BLEU Score vs. Pruning Percentage (BERT)')
plt.xlabel('Pruning Percentage (%)')
plt.ylabel('BLEU Score')
plt.xticks(prune_percents)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('bleu_score_vs_pruning_bert.png')

# Show the plot
plt.show()
