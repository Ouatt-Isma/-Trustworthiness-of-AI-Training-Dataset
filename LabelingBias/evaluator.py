import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})


class TrustOp:
    def __init__(self, t, d, u, a=0.5):

        assert round(t+d+u,10)==1, f"t={t}, d={d}, u={u}"
        self.t = t
        self.d = d
        self.a = a 
        self.u = 1 - self.t - self.d
        

data= np.load("./cifar10h-counts.npy")
def compute_stats(data):
    q1 = np.percentile(data, 25)
    median = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_whisker = np.min(data[data >= q1 - 1.5 * iqr])
    upper_whisker = np.max(data[data <= q3 + 1.5 * iqr])
    return lower_whisker, q1, median, q3, upper_whisker

def plot(datab, datad, datau, title):
    stats1 = compute_stats(datab)
    stats2 = compute_stats(datad)
    stats3 = compute_stats(datau)
# Create the box plot
    fig, ax = plt.subplots()
    labels=['Trust Mass', 'Distrust Mass', 'Uncertainty Mass']
    box = ax.boxplot([datab, datad, datau], vert=True, patch_artist=True, labels=labels)

    # Define colors
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add inline annotations for each box
    positions = [1, 2, 3]
    groups = [stats1, stats2, stats3]
    ticks = []
    for pos, stats in zip(positions, groups[:2]):
        lw, q1, med, q3, uw = stats
        # ax.text(pos + 0.1, lw, f"Min: {lw:.2f}", color='blue', fontsize=8, va='center')
        # ax.text(pos + 0.1, q1, f"Q1: {q1:.2f}", color='green', fontsize=8, va='center')
        # ax.text(pos + 0.1, med, f"Median: {med:.2f}", color='red', fontsize=8, va='center')
        # ax.text(pos + 0.1, q3, f"Q3: {q3:.2f}", color='green', fontsize=8, va='center')
        # ax.text(pos + 0.1, uw, f"Max: {uw:.2f}", color='blue', fontsize=8, va='center')
        # ticks += [lw, q1, med, q3, uw]
        ticks += [lw, uw]
        
    

    # Combine existing y-ticks with custom ticks, sort them and remove duplicates
    print(ticks)
    all_ticks = sorted(set(ax.get_yticks()).union(ticks))
    ax.set_yticks(all_ticks)
    # Add legend manually
    for i, color in enumerate(colors):
        ax.plot([], [], color=color, label=labels[i])
    # ax.legend(loc='upper right')

    # Clean plot
    ax.set_title(title)
    ax.set_ylabel("Values")
    plt.tight_layout()
    plt.savefig(f"{title}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()
    
# def plot(data):
    
#     q1 = np.percentile(data, 25)
#     median = np.percentile(data, 50)
#     q3 = np.percentile(data, 75)
#     iqr = q3 - q1
#     lower_whisker = np.min(data[data >= q1 - 1.5 * iqr])
#     upper_whisker = np.max(data[data <= q3 + 1.5 * iqr])

#     # Create a box plot
#     fig, ax = plt.subplots()
#     ax.boxplot([data,data], vert=True, patch_artist=True,labels=['Group A', 'Group B', 'Group C'])
#     # Add notes directly in the plot area for each key statistic
#     ax.text(1.1, lower_whisker, f"Min: {lower_whisker:.2f}", color='blue', fontsize=9, verticalalignment='center')
#     ax.text(1.1, q1, f"Q1: {q1:.2f}", color='green', fontsize=9, verticalalignment='center')
#     ax.text(1.1, median, f"Median: {median:.2f}", color='red', fontsize=9, verticalalignment='center')
#     ax.text(1.1, q3, f"Q3: {q3:.2f}", color='green', fontsize=9, verticalalignment='center')
#     ax.text(1.1, upper_whisker+0.01, f"Max: {upper_whisker:.2f}", color='blue', fontsize=9, verticalalignment='center')




#     # Set plot title and labels
#     ax.set_title("Vertical Box Plot with Statistical Ticks")
#     ax.set_ylabel("Values")

#     plt.tight_layout()
#     plt.show()


def method_2(data=data, labels=None):
    res = []
    labels_to = []
    for i in range(len(data)):
        no = np.sum(data[i])
        # assert no==10, no
        if(not labels):
            r = np.max(data[i])
            labels_to.append(np.argmax(data[i]))
        else:
            r = data[i][labels[i]]
        s = no-r
        W = 2
        tot = r+s+W
        op= TrustOp(r/tot, s/tot, W/tot)
        res.append(op)
        
            
    return res, labels_to



res,labels = method_2()
print(len(res))
datab = np.array([a.t for a in res])
datad =np.array([a.d for a in res])
datau = np.array([a.u for a in res])
title = "CIFAR 10H"
plot(datab, datad, datau, title)

def normalize_row_sum_to_10(arr):
    arr_new = arr.copy()
    for i in range(arr.shape[0]):
        row = arr_new[i]
        total = np.sum(row)
        if total <= 10:
            continue  # already fine

        excess = total - 10
        # Get descending order of indices
        idx_sorted = np.argsort(-row)
        
        for idx in idx_sorted:
            reducible = min(row[idx], excess)
            row[idx] -= reducible
            excess -= reducible
            if excess <= 0:
                break
        arr_new[i] = row
    return arr_new


arr_new = normalize_row_sum_to_10(data)
res,_ = method_2(arr_new, labels)
print(len(res))
datab = np.array([a.t for a in res])
datad =np.array([a.d for a in res])
datau = np.array([a.u for a in res])
title = "CIFAR 10H Cropped to 10 annotators"
plot(datab, datad, datau, title)


# myd = [a.t for a in res]
# print(myd)



import pandas as pd

def compute_disagreement(row):
    """Disagreement = 1 - proportion of majority label"""
    total = np.sum(row)
    maj = np.max(row)
    return 1 - maj / total if total > 0 else 0

def subsample_annotations(data, n):
    """
    Subsample each row of data to exactly n annotations 
    while preserving majority label.
    """
    arr_new = []
    for row in data:
        total = np.sum(row)
        if total <= n:
            arr_new.append(row)
            continue
        
        # Subsample proportional to original counts but keep majority label
        maj_idx = np.argmax(row)
        counts = row.copy().astype(int)
        
        # Force majority label to keep majority after subsample
        maj_keep = max(1, int(np.round((row[maj_idx] / total) * n)))
        remaining = n - maj_keep
        
        # Allocate remaining proportionally to other labels
        others = np.delete(np.arange(len(row)), maj_idx)
        if remaining > 0:
            proportions = row[others] / np.sum(row[others]) if np.sum(row[others]) > 0 else np.zeros(len(others))
            alloc = np.random.multinomial(remaining, proportions)
            new_row = np.zeros_like(row, dtype=int)
            new_row[maj_idx] = maj_keep
            new_row[others] = alloc
        else:
            new_row = np.zeros_like(row, dtype=int)
            new_row[maj_idx] = maj_keep
        arr_new.append(new_row)
    return np.array(arr_new)

def run_experiment_u(data, n, W=2):
    """
    Subsample annotations to n per item,
    compute SL uncertainty and disagreement,
    then compute correlation and average u.
    """
    subsampled = subsample_annotations(data, n)
    res, _ = method_2(subsampled)
    us = np.array([a.u for a in res])
    disagreements = np.array([compute_disagreement(row) for row in subsampled])
    
    corr = np.corrcoef(us, disagreements)[0,1]
    avg_u = np.mean(us)
    return corr, avg_u

def run_experiment_t(data, n, W=2):
    """
    Subsample annotations to n per item,
    compute SL uncertainty and disagreement,
    then compute correlation and average u.
    """
    subsampled = subsample_annotations(data, n)
    res, _ = method_2(subsampled)
    us = np.array([a.t for a in res])
    disagreements = np.array([compute_disagreement(row) for row in subsampled])
    
    corr = np.corrcoef(us, disagreements)[0,1]
    avg_t = np.mean(us)
    return corr, avg_t

def run_experiment_d(data, n, W=2):
    """
    Subsample annotations to n per item,
    compute SL uncertainty and disagreement,
    then compute correlation and average u.
    """
    subsampled = subsample_annotations(data, n)
    res, _ = method_2(subsampled)
    us = np.array([a.d for a in res])
    disagreements = np.array([compute_disagreement(row) for row in subsampled])
    
    corr = np.corrcoef(us, disagreements)[0,1]
    avg_d = np.mean(us)
    return corr, avg_d

def run_experiment(data, n, W=2):
    """
    Subsample annotations to n per item,
    compute SL uncertainty and disagreement,
    then compute correlation and average u.
    """
    subsampled = subsample_annotations(data, n)
    res, _ = method_2(subsampled)
    us = np.array([a.u for a in res])
    ts = np.array([a.u for a in res])
    ds = np.array([a.u for a in res])
    disagreements = np.array([compute_disagreement(row) for row in subsampled])
    
    corr_u = np.corrcoef(us, disagreements)[0,1]
    avg_u = np.mean(us)
    
    corr_t = np.corrcoef(ts, disagreements)[0,1]
    avg_t = np.mean(ts)
    
    corr_d = np.corrcoef(ds, disagreements)[0,1]
    avg_d = np.mean(ds)
    return {"t": (corr_t, avg_t), "d": (corr_d, avg_d),"u": (corr_u, avg_u)}


# Run for 5, 10, 20, 50 annotations
results = []
for n in [5, 10, 20, 50]:
    stats = run_experiment(data, n)
    results.append([
        n,
        round(stats["t"][0], 2), round(stats["t"][1], 2),
        round(stats["d"][0], 2), round(stats["d"][1], 2),
        round(stats["u"][0], 2), round(stats["u"][1], 2),
    ])

# Build DataFrame
cols = [
    "Annotations per item",
    "Corr(t)", "Avg(t)",
    "Corr(d)", "Avg(d)",
    "Corr(u)", "Avg(u)"
]
df = pd.DataFrame(results, columns=cols)
print(df)

# Optional: save nicely
df.to_latex("results_table.tex", index=False, float_format="%.2f")
df.to_csv("results_table.csv", index=False)