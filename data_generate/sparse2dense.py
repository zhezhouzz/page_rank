"""Sparse2Dense

   by Zhou Zhe"""

#!/usr/local/bin/python3

from optparse import OptionParser
import numpy as np
import scipy.io as sio


def main():
    """main func of page rank"""
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="output",
                      help="write report to output", metavar="output")
    parser.add_option("-i", "--input", dest="input",
                      help="matrix data input", metavar="input")
    (options, args) = parser.parse_args()

    mm_page_map = sio.mmread(options.input)
    # mm_page_map = sio.mmread("../data/page_map.mtx")
    print(mm_page_map.A)
    page_map = np.array(mm_page_map.A)
    # sio.mmwrite("../data/page_map_dense.mtx", page_map)
    sio.mmwrite(options.output, page_map)


if __name__ == "__main__":
    main()
