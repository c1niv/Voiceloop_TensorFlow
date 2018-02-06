import pickle
import numpy as np
from utils_python2 import generate_merlin_wav

def main():
    files = pickle.load(open("tmp_out.pckl", "rb"))
    for data in files:
        generate_merlin_wav(data[0],
                            data[1],
                            data[2],
                            data[3])

if __name__ == '__main__':
    main()
