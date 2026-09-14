data = [1, 2, 2, 3, 4, 4, 4, 5]
# 1. 移除重複
unique_data = list(set(data))

# 2. 計算次數
counts = {}
for item in data:
    counts[item] = counts.get(item, 0) + 1

print(f"Unique: {unique_data}")
print(f"Counts: {counts}")
