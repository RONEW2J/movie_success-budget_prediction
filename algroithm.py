import random
import time

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False 
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

print("Размер массива | Время выполнения (сек)")
print("-" * 40)

for n in sizes:
    arr = [random.randint(0, 10000) for _ in range(n)]
    
    start_time = time.time()
    bubble_sort(arr)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"{n:<15} | {execution_time:.4f}")