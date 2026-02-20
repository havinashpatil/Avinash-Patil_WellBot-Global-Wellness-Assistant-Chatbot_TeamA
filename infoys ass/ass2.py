import gc

# 1. enumerate()
print("Enumerate Example:")
fruits = ["apple", "banana", "mango"]
for index, value in enumerate(fruits, start=1):
    print(index, value)

# 2. isinstance()
print("\nIsinstance Example:")
x = 25.5
print("Is x float?", isinstance(x, float))
print("Is x int or float?", isinstance(x, (int, float)))

# 3. Garbage Collection
print("\nGarbage Collection Example:")
print("Before:", gc.get_count())
gc.collect()
print("After:", gc.get_count())

# Example of gc module

import gc

# Show garbage collection counts
print("GC count before collection:", gc.get_count())

# Manually trigger garbage collection
collected = gc.collect()
print("Number of objects collected:", collected)

print("GC count after collection:", gc.get_count())

# Disable and enable garbage collection
gc.disable()
print("Garbage collection disabled")

gc.enable()
print("Garbage collection enabled")

# Example of isinstance()

x = 10
y = 10.5
z = "Hello"

print(isinstance(x, int))        # True
print(isinstance(y, float))      # True
print(isinstance(z, str))        # True

# Checking multiple types
print(isinstance(y, (int, float)))   # True

# Example of enumerate()

fruits = ["apple", "banana", "mango"]

# Default start = 0
for index, value in enumerate(fruits):
    print(index, value)

print("\nWith custom start value:")

# Custom start value
for index, value in enumerate(fruits, start=1):
    print(index, value)