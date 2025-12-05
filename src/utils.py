import numpy as np
from src.dataset import load_dataset

# Load mapping once at module level
_, _, mapping, _ = load_dataset()

def label_to_emoji(label):
    label_emoji_mapping = dict(zip(mapping['number'], mapping['emoticons']))
    return label_emoji_mapping.get(label, "❓")

def labels_to_emojis(labels):
    return [label_to_emoji(label) for label in labels]

# --- ADDED FOR EASTER EGG EXPERIMENT ---
def inject_noise(y_labels, noise_rate=0.2, num_classes=20, seed=42):
    """
    Randomly flips 'noise_rate' percentage of labels to a random different class.
    Used to test model robustness against noisy/subjective data.
    """
    if noise_rate <= 0:
        return y_labels
        
    # Fix seed for reproducibility
    np.random.seed(seed)
    
    y_noisy = np.copy(y_labels)
    n_samples = len(y_labels)
    n_noisy = int(n_samples * noise_rate)
    
    # Choose random indices to flip
    noise_indices = np.random.choice(n_samples, n_noisy, replace=False)
    
    # Assign random new labels (different from true label)
    for idx in noise_indices:
        true_label = y_labels[idx]
        possible_labels = list(range(num_classes))
        if true_label in possible_labels:
            possible_labels.remove(true_label)
        y_noisy[idx] = np.random.choice(possible_labels)
        
    print(f"⚠️ Easter Egg: Injected {noise_rate*100:.0f}% noise into training labels!")
    return y_noisy