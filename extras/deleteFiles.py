import os
import sys

#def main():

if __name__ == '__main__':
    file = 'extras/files_to_delete.txt'
    base_files = 'results'

    with open(file) as f: 
        for line in f: 
            for elem in line.split('\t'):
                try:
                    os.remove(os.path.join(base_files, elem.strip()))
                except:
                    print("Error:", sys.exc_info()[0])
