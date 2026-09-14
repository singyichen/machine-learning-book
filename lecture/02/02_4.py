import random

nums = [random.randint(1, 100) for _ in range(5)]
print(f"List: {nums}")
print(f"Max: {max(nums)}, Min: {min(nums)}, Sum: {sum(nums)}")
