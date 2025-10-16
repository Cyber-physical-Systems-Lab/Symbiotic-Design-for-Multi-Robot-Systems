import numpy as np

# Function to calculate the time for each leg of the trip
def calculate_time(path):
    # Time is simply the number of steps (frames) the car takes
    return len(path)  # As speed is assumed to be 1 unit per frame

# Calculate time for both small and big cars
time_small_pickup = calculate_time(small_path_pickup)
time_small_delivery = calculate_time(small_path_delivery)
time_small_return = calculate_time(small_path_return)

time_big_pickup = calculate_time(big_path_pickup)
time_big_delivery = calculate_time(big_path_delivery)
time_big_return = calculate_time(big_path_return)

# Total time taken for each car
total_time_small = time_small_pickup + time_small_delivery + time_small_return
total_time_big = time_big_pickup + time_big_delivery + time_big_return

# Print the times for both cars
print(f"Time taken by Small Car to pickup, deliver, and return: {total_time_small} units")
print(f"Time taken by Big Car to pickup, deliver, and return: {total_time_big} units")
