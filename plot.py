from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Apply smoothing with a window size of 50
def smooth(losses, window_size=50):
    kernel = np.ones(window_size) / window_size
    return np.convolve(losses, kernel, mode='valid')

def get_data_from_file(f: Path):
    lines = f.read_text().splitlines()
    train_losses = [eval(l[l.index('['):]) for l in lines if 'train_losses=[' in l]
    val_losses = [eval(l[l.index('{'):]) for l in lines if 'val_losses={' in l]
    assert len(train_losses) == 1
    assert len(val_losses) == 1
    return {"train_losses": train_losses[0], "val_losses": val_losses[0]}

ROOT = Path(__file__).parent
RUNS = [
    {"name": "Softmax", "color": "blue", **get_data_from_file(ROOT / "experiment_logs/20250316_Softmax.txt")},
    {"name": "Dynamic Tanh", "color": "red", **get_data_from_file(ROOT / "experiment_logs/20250316_DynamicTanh.txt")},
]

plt.figure(figsize=(16, 10))

# Plot training losses (faded lines)
for run in RUNS:
    smooth_train = smooth(run["train_losses"])
    plt.plot(smooth_train, alpha=0.3, label=f'{run["name"]} (train, smoothed)', color=run["color"])
    val_steps = run["val_losses"].keys()
    val_values = run["val_losses"].values()
    plt.plot(val_steps, val_values, label=f'{run["name"]} (val)', color=run["color"])

# Plot validation losses (solid points)
plt.ylim(3.7, 4.2)

plt.xlabel('Steps (*16k tokens)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("dynamic_tanh.png")


if False:
    ## Analyse fineweb seq_len distribution
    # Load validation data
    val_data = np.fromfile('./data/fineweb1B/fineweb_val_000000.bin', dtype=np.uint16)

    # Find document boundaries (50256) and calculate sequence lengths
    doc_ends = np.flatnonzero(val_data == 50256)
    seq_lengths = np.diff(doc_ends)

    # Create histogram
    plt.figure(figsize=(16, 10))
    plt.hist(seq_lengths, bins=np.geomspace(128, 128*1024, 21), alpha=0.7, log=True)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Document Lengths')
    plt.grid(True, alpha=0.3)
    plt.gca().set_xscale("log", base=2)
    plt.gca().set_yscale("linear")
    plt.savefig("seq_lengths_hist.png")
