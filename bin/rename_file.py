#!/usr/bin/python

import glob
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument("prefix")
args = parser.parse_args()
prefix = args.prefix

for name in glob.glob('*.png'):
    os.rename(name, prefix + '_' + name)