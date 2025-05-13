import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

# Load the data
url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv"
df = pd.read_csv(url)

# Preview the data
print(df.head())

# Count of players
print("Number of players:", df.userid.nunique())
print("Number of total records:", len(df.userid))

# Convert boolean to integer
df['retention_1'] = df['retention_1'].astype(int)
df['retention_7'] = df['retention_7'].astype(int)

# Group data
control = df[df['version'] == 'gate_30']
treatment = df[df['version'] == 'gate_40']

# Summary info
print(df.info())
print("\nVersion distribution:")
print(df['version'].value_counts())
print("\nOverall retention means:")
print(df[['retention_1', 'retention_7']].mean())

# Group-wise retention rates
retention_1_rates = df.groupby('version')['retention_1'].mean() * 100
retention_7_rates = df.groupby('version')['retention_7'].mean() * 100

print("\n1-day retention rates (%):")
print(retention_1_rates)
print("\n7-day retention rates (%):")
print(retention_7_rates)

# Posterior distribution setup for 1-day retention
successes_30 = control['retention_1'].sum()
trials_30 = control['retention_1'].count()
successes_40 = treatment['retention_1'].sum()
trials_40 = treatment['retention_1'].count()

x = np.linspace(0, 0.6, 1000)
posterior_30 = beta(successes_30 + 1, trials_30 - successes_30 + 1)
posterior_40 = beta(successes_40 + 1, trials_40 - successes_40 + 1)

# Plot posterior distributions for 1-day retention
plt.figure(figsize=(10, 5))
plt.plot(x, posterior_30.pdf(x), label='gate_30')
plt.plot(x, posterior_40.pdf(x), label='gate_40')
plt.xlabel('Retention Rate')
plt.ylabel('Density')
plt.title('Posterior Distributions for 1-Day Retention')
plt.legend()
plt.show()

# Simulate samples from the Beta posteriors
samples_30 = np.random.beta(successes_30 + 1, trials_30 - successes_30 + 1, 100000)
samples_40 = np.random.beta(successes_40 + 1, trials_40 - successes_40 + 1, 100000)
diff = samples_30 - samples_40

# Plot difference in posteriors (1-day)
plt.hist(diff, bins=50, color='purple', alpha=0.7)
plt.title('Posterior Distribution of the Difference (gate_30 - gate_40)\n1-Day Retention')
plt.xlabel('Difference in Retention Rate')
plt.axvline(0, color='black', linestyle='--')
plt.ylabel('Frequency')
plt.show()

# Probability that gate_30 is better than gate_40 for 1-day retention
prob_1day = (diff > 0).mean()
print(f"\nProbability that gate_30 has better 1-day retention: {prob_1day:.2%}")

# Posterior distribution setup for 7-day retention
successes_30_ret7 = control['retention_7'].sum()
trials_30_ret7 = control['retention_7'].count()
successes_40_ret7 = treatment['retention_7'].sum()
trials_40_ret7 = treatment['retention_7'].count()

samples_30_ret7 = np.random.beta(successes_30_ret7 + 1, trials_30_ret7 - successes_30_ret7 + 1, 100000)
samples_40_ret7 = np.random.beta(successes_40_ret7 + 1, trials_40_ret7 - successes_40_ret7 + 1, 100000)
diff_ret7 = samples_30_ret7 - samples_40_ret7

# Plot difference in posteriors (7-day)
plt.hist(diff_ret7, bins=50, color='green', alpha=0.7)
plt.title('Posterior Distribution of the Difference (gate_30 - gate_40)\n7-Day Retention')
plt.xlabel('Difference in Retention Rate')
plt.axvline(0, color='black', linestyle='--')
plt.ylabel('Frequency')
plt.show()

# Probability that gate_30 is better than gate_40 for 7-day retention
prob_7day = (diff_ret7 > 0).mean()
print(f"\nProbability that gate_30 has better 7-day retention: {prob_7day:.2%}")

# Retention by group
prop_gate30_1 = (control['retention_1'].sum() / len(control)) * 100
prop_gate40_1 = (treatment['retention_1'].sum() / len(treatment)) * 100

print(f"\n1-Day Retention:\nGate 30: {prop_gate30_1:.2f}%\nGate 40: {prop_gate40_1:.2f}%")

prop_gate30_7 = (control['retention_7'].sum() / len(control)) * 100
prop_gate40_7 = (treatment['retention_7'].sum() / len(treatment)) * 100

print(f"\n7-Day Retention:\nGate 30: {prop_gate30_7:.2f}%\nGate 40: {prop_gate40_7:.2f}%")

# Overall retention
overall_ret1 = (df['retention_1'].sum() / len(df)) * 100
overall_ret7 = (df['retention_7'].sum() / len(df)) * 100

print(f"\nOverall Retention:\n1-Day: {overall_ret1:.2f}%\n7-Day: {overall_ret7:.2f}%")

# Bar plot for retention by version
means = df.groupby('version')[['retention_1', 'retention_7']].mean().reset_index()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].bar(means['version'], means['retention_1'], color=['skyblue', 'orange'])
ax[0].set_title('1-Day Retention Rate')
ax[0].set_ylabel('Retention Rate')
ax[0].set_ylim(0, 0.5)

ax[1].bar(means['version'], means['retention_7'], color=['skyblue', 'orange'])
ax[1].set_title('7-Day Retention Rate')
ax[1].set_ylim(0, 0.25)

plt.tight_layout()
plt.show()
