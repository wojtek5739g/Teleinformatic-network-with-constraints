import os, sys
import csv

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def save_data(data, unique_filename):
    path = f'./logs/{unique_filename}.txt'
    exists = os.path.exists(path)

    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)

        if not exists:
            writer.writerow(["Iteration", "Avg", "Max", "Min"])

        writer.writerow([data[0], data[1], data[2], data[3]])