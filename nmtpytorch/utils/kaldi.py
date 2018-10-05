import os
import struct
import functools

import numpy

ERROR_BINARY = "Binary mode header ('\0B') not found when reading a matrix."
ERROR_READ_MAT = "Unknown matrix format '{}'. Supported ones: DM(float64), FM(float32)."
ERROR_WRITE_MAT = "Unknown matrix format '{}'. Supported ones are float64, float32."


def readString(f):
    s = ""
    while True:
        c = f.read(1).decode('utf-8')
        if c == "":
            raise ValueError("EOF encountered while reading a string.")
        if c == " ":
            return s
        s += c


def readInteger(f):
    n = ord(f.read(1))
    a = f.read(n)[::-1]
    try:
        return int.from_bytes(a, byteorder='big', signed=False)
    except Exception as e:
        return functools.reduce(lambda x, y: x * 256 + ord(y), a, 0)


def readMatrix(f):
    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError(ERROR_BINARY)
    mat_format = readString(f)
    nRows = readInteger(f)
    nCols = readInteger(f)
    if mat_format == "DM":
        data = struct.unpack("<%dd" % (nRows * nCols), f.read(nRows * nCols * 8))
        data = numpy.array(data, dtype="float64")
    elif mat_format == "FM":
        data = struct.unpack("<%df" % (nRows * nCols), f.read(nRows * nCols * 4))
        data = numpy.array(data, dtype="float32")
    else:
        raise ValueError(ERROR_READ_MAT.format(mat_format))
    return data.reshape(nRows, nCols)


def readMatrixShape(f):
    header = f.read(2).decode('utf-8')
    if header != "\0B":
        raise ValueError(
            "Binary mode header ('\0B') not found when attempting to read a matrix.")
    mat_format = readString(f)
    nRows = readInteger(f)
    nCols = readInteger(f)
    if mat_format == "DM":
        f.seek(nRows * nCols * 8, os.SEEK_CUR)
    elif mat_format == "FM":
        f.seek(nRows * nCols * 4, os.SEEK_CUR)
    else:
        raise ValueError(ERROR_READ_MAT.format(mat_format))
    return nRows, nCols


def writeString(f, s):
    f.write((s + " ").encode('utf-8'))


def writeInteger(f, a):
    s = struct.pack("<i", a)
    f.write(chr(len(s)).encode('utf-8') + s)


def writeMatrix(f, data):
    f.write('\0B'.encode('utf-8'))      # Binary data header
    if str(data.dtype) == "float64":
        writeString(f, "DM")
        writeInteger(f, data.shape[0])
        writeInteger(f, data.shape[1])
        f.write(struct.pack("<%dd" % data.size, *data.ravel()))
    elif str(data.dtype) == "float32":
        writeString(f, "FM")
        writeInteger(f, data.shape[0])
        writeInteger(f, data.shape[1])
        f.write(struct.pack("<%df" % data.size, *data.ravel()))
    else:
        raise ValueError(ERROR_WRITE_MAT.format(str(data.dtype)))


def readArk(filename, limit=numpy.inf):
    """
    Reads the features in a Kaldi ark file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []
    uttids = []
    with open(filename, "rb") as f:
        while True:
            try:
                uttid = readString(f)
            except ValueError:
                break
            feature = readMatrix(f)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit:
                break
    return features, uttids


def readMatrixByOffset(arkfile, offset):
    with open(arkfile, "rb") as g:
        g.seek(offset)
        feature = readMatrix(g)
    return feature


def readScp(filename, limit=numpy.inf):
    """
    Reads the features in a Kaldi script file.
    Returns a list of feature matrices and a list of the utterance IDs.
    """
    features = []
    uttids = []
    with open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p + 1:])
            with open(arkfile, "rb") as g:
                g.seek(offset)
                feature = readMatrix(g)
            features.append(feature)
            uttids.append(uttid)
            if len(features) == limit:
                break
    return features, uttids


def read_scp_info(filename, limit=numpy.inf):
    res = []
    with open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p + 1:])
            with open(arkfile, "rb") as g:
                g.seek(offset)
                feat_len, feat_dim = readMatrixShape(g)
            res.append((uttid, arkfile, offset, feat_len, feat_dim))
            if len(res) == limit:
                break
    return res


def read_scp_info_dic(filename, limit=numpy.inf):
    res = {}
    with open(filename, "r") as f:
        for line in f:
            uttid, pointer = line.strip().split()
            p = pointer.rfind(":")
            arkfile, offset = pointer[:p], int(pointer[p + 1:])
            with open(arkfile, "rb") as g:
                g.seek(offset)
                feat_len, feat_dim = readMatrixShape(g)
            res[uttid] = ((uttid, arkfile, offset, feat_len, feat_dim))
            if len(res) == limit:
                break
    return res


def writeArk(filename, features, uttids):
    """
    Takes a list of feature matrices and a list of utterance IDs,
      and writes them to a Kaldi ark file.
    Returns a list of strings in the format "filename:offset",
      which can be used to write a Kaldi script file.
    """
    pointers = []
    with open(filename, "ab") as f:
        for feature, uttid in zip(features, uttids):
            writeString(f, uttid)
            pointers.append("%s:%d" % (filename, f.tell()))
            writeMatrix(f, feature)
    return pointers


def writeScp(filename, uttids, pointers):
    """
    Takes a list of utterance IDs and a list of strings in the format "filename:offset",
      and writes them to a Kaldi script file.
    """
    with open(filename, "w") as f:
        for uttid, pointer in zip(uttids, pointers):
            f.write("%s %s\n" % (uttid, pointer))
