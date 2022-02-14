import string
import random as rd
from argparse import ArgumentTypeError


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def split_parser(val):
    """Parse float numbers"""
    try:
        val = float(val)
    except ValueError:
        raise ArgumentTypeError(f'{val} must be float')
    if val < 0.0 or val > 1.0:
        raise ArgumentTypeError('Split out of bounds (0,1)')
    return val


def zim_annotations_list(annotations):
    """Given an annotation file, return a list of abbraviated annotations."""
    result = []
    with open(annotations, 'r') as ann:
        for line in ann:
            result.append('_'.join(
                [s[0:2] for s in line.split(' ', 1)[1].strip().split(' ')]))
    return result


def zim_label_to_number(annotations):
    """Given an annotation file, return key-value dict of abbraviated ones."""
    labels = zim_annotations_list(annotations)
    result = {}
    for i, label in enumerate(labels):
        result[label] = i
    return result


def label_to_number(annotations):
    """Given an annotation file, return key-value dict of full-names"""
    result = {}
    with open(annotations, 'r') as ann:
        for line in ann:
            tup = line.split(' ')
            result[tup[1].strip()] = tup[0]
    return result

