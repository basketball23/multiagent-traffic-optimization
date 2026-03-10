import csv

filename = "results_conn1_ep10.csv"
column_name = "system_total_waiting_time"

total = 0
count = 0

with open(filename, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        value = float(row[column_name])
        total += value
        count += 1

average = total / count if count > 0 else 0
print("Average:", average)
