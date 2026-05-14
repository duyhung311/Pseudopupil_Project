import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your data
# Replace 'your_file.csv' with your actual filename
df = pd.read_csv('training_log.csv')

# 2. Create the plot
plt.figure(figsize=(10, 6))

# Plot training loss
plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue', linewidth=2)

# Plot validation loss
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linewidth=2)

# 3. Add details
plt.title('Training and Validation Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 4. Show the plot
plt.show()