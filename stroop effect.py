# %%
import pandas as pd
import numpy as np
from scipy.stats import t

import seaborn as sns

# %%
df = pd.read_csv('stroopdata.csv')

dfstat = pd.concat([
    (df.describe()
     .rename({'50%': 'median'})
     .loc[['count', 'mean', 'median', 'std'], :]
     ),
    (df.sem()
     .to_frame()
     .rename(columns={0: 'sem'})
     .T
     )
])

%matplotlib inline
df.plot(kind='hist', bins=20, alpha=0.5)

# %%
diff = df.Congruent - df.Incongruent
diff_mean = diff.mean()

# computed using a 'manual' formula for debugging
n = len(diff)
diff_sem = np.sqrt(((diff - diff_mean)**2).sum() / (n - 1)) / np.sqrt(n)
# computed using a function from a library
diff_sem = diff.sem()

t_statistic = diff_mean / diff_sem

# 1-tail t-test
ddof = n - 1
alpha = 0.05

# read 'manually' from a table for debugging
t_critical = -1.714
# computed using a function from a library
t_critical = t.ppf(alpha, ddof)

print('t critical value: {:.3f} (for alpha={})'.format(t_critical, alpha))
print('t statistic: {:.3f}'.format(t_statistic))
print('null hypothesis {}\n'.format('rejected' if t_statistic < t_critical else 'accepted'))

# 95% confidence interval
t_95 = t.ppf(1 - 0.025, ddof)
confidence95 = (diff_mean - t_95 * diff_sem, diff_mean + t_95 * diff_sem)
print('''95% confidence interval: difference between congruent and incongruent reading time is
between {:.2f}s and {:.2f}s with 95% confidence.
'''.format(*confidence95))

p_value = t.cdf(t_statistic, ddof)
print('p value: {:.3e}'.format(p_value))
r2 = t_statistic**2 / (t_statistic**2 + ddof)
print('R2: {:.3f}'.format(r2))

# %%
