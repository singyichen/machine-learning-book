# 1. Import Module
import matplotlib.pyplot as plt

# 2. Prepare Data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 3. Create a Line Plot
plt.plot(x, y, marker='o', label='Trend Line')

# 4. Customization
plt.title('Demo of Line Plot')    # Title
plt.xlabel('X-axis Label')     # X-axis label
plt.ylabel('Y-axis Label')     # Y-axis label
plt.legend()                   # Show legend
plt.grid(True)                 # Enable grid

# 5. Display the plot
plt.show()





