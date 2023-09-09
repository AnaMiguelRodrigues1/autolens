# Creating folder resources to store results, models and more
import os

def resources():
    folder_name = './resources'

    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
        except OSError as e:
            print(f"Error: {e}")
    else:
        print('')
