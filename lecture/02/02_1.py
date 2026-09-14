import math

r_str = input("Enter radius: ")
try:
    radius = float(r_str)
    area = math.pi * (radius ** 2)
    print(f"The area of a circle with radius {radius} is {area:.2f}")
except ValueError:
    print("Please enter a valid number.")
