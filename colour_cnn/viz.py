import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
df = pd.read_csv('finaldata2.csv')
# Assuming your DataFrame is named 'df'
# Assuming your DataFrame columns are named 'Prediction', 'R', and 'B'

# Filter rows where Prediction is "Blue"
blue_rows = df[df['Prediction'] == 'Blue']

# Filter rows where Prediction is not "Blue"
non_blue_rows = df[df['Prediction'] != 'Blue']

# Creating a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the line graph for blue rows
blue_scatter = axes[0].plot(blue_rows['B'], blue_rows['G'], 'o', label='Blue Prediction')
axes[0].plot(blue_rows['R'], blue_rows['R'], label='45 Degree Line')
axes[0].set_xlabel('B')
axes[0].set_ylabel('G')
axes[0].legend()

# Plotting the line graph for non-blue rows
non_blue_scatter = axes[1].plot(non_blue_rows['B'], non_blue_rows['G'], 'o', color='red', label='Non-Blue Prediction')
axes[1].plot(non_blue_rows['R'], non_blue_rows['R'], color='red', label='45 Degree Line')
axes[1].set_xlabel('B')
axes[1].set_ylabel('G')
axes[1].legend()

# Adding cursor hover functionality to blue rows plot
mplcursors.cursor(blue_scatter).connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f})"))

# Adding cursor hover functionality to non-blue rows plot
mplcursors.cursor(non_blue_scatter).connect("add", lambda sel: sel.annotation.set_text(f"({sel.target[0]:.2f}, {sel.target[1]:.2f})"))

# Displaying the plot
plt.tight_layout()
plt.show()
