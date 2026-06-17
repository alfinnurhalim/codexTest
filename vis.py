import numpy as np
import matplotlib.pyplot as plt

# =========================
# Dummy data
# =========================
np.random.seed(42)

source = np.random.normal(loc=50, scale=10, size=5000)
target = np.random.normal(loc=70, scale=15, size=5000)

# optional clipping, like real bounded values
source = np.clip(source, 0, 100)
target = np.clip(target, 0, 100)


# =========================
# Histogram / distribution matching
# =========================
def match_distribution(source, target):
    """
    Match source distribution to target distribution using quantile mapping.
    """
    source_sorted = np.sort(source)
    target_sorted = np.sort(target)

    source_quantiles = np.linspace(0, 1, len(source_sorted))
    target_quantiles = np.linspace(0, 1, len(target_sorted))

    # Convert each source value to its quantile
    source_value_to_quantile = np.interp(
        source,
        source_sorted,
        source_quantiles
    )

    # Convert that quantile to target value
    matched = np.interp(
        source_value_to_quantile,
        target_quantiles,
        target_sorted
    )

    return matched


matched = match_distribution(source, target)


# =========================
# Print stats
# =========================
def print_stats(name, data):
    print(f"{name}")
    print(f"  mean   : {np.mean(data):.2f}")
    print(f"  median : {np.median(data):.2f}")
    print(f"  std    : {np.std(data):.2f}")
    print(f"  min    : {np.min(data):.2f}")
    print(f"  max    : {np.max(data):.2f}")
    print()


print_stats("Source", source)
print_stats("Target", target)
print_stats("Matched", matched)


# =========================
# Visualization
# =========================
bins = np.arange(0, 101, 2)

plt.figure(figsize=(12, 6))

plt.hist(source, bins=bins, alpha=0.5, density=True, label="Source")
plt.hist(target, bins=bins, alpha=0.5, density=True, label="Target")
plt.hist(matched, bins=bins, alpha=0.5, density=True, label="Matched")

plt.title("Histogram Matching with Dummy Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()