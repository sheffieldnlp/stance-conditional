import io
import subprocess
import sys
import numpy as np


def printPredsToFile(infile, infileenc, outfile, res, skip=0):
    """
    Print predictions to file in SemEval format so the official eval script can be applied
    :param infile: official stance data for which predictions are made
    :param infileenc: encoding of the official stance data file
    :param outfile: file to print to
    :param res: python list of results. 0 for NONE predictions, 1 for AGAINST predictions, 2 for FAVOR
    :param skip: how many testing instances to skip from the beginning, useful if part of the file is used for dev instead of test
    """
    outf = open(outfile, 'w')
    cntr = 0
    for line in io.open(infile, encoding=infileenc, mode='r'): #for the unlabelled Trump dev file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n"))
        elif skip > 0:
            skip -= 1
        else:
            outl = line.strip("\n").split("\t")
            if res[cntr] == 0:
                outl[3] = 'NONE'
            elif res[cntr] == 1:
                outl[3] = 'AGAINST'
            elif res[cntr] == 2:
                outl[3] = 'FAVOR'
            outf.write("\n" + "\t".join(outl))
            cntr += 1

    outf.close()



def printPredsToFileByID(infile, outfile, ids, res, infileenc='windows-1252'):
    """
    Print predictions to file in SemEval format so the official eval script can be applied
    :param infile: official stance data for which predictions are made
    :param infileenc: encoding of the official stance data file
    :param outfile: file to print to
    :param res: python list of results. 0 for NONE predictions, 1 for AGAINST predictions, 2 for FAVOR
    :param skip: how many testing instances to skip from the beginning, useful if part of the file is used for dev instead of test
    """
    outf = open(outfile, 'w')
    for line in io.open(infile, encoding=infileenc, mode='r'): #for the unlabelled Trump dev file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n"))
        else:
            outl = line.strip("\n").split("\t")
            realid = np.asarray(outl[0], dtype=np.float64)
            cnt = 0
            for tid in ids:
                if tid == realid:
                    idx = cnt
                    #print("found!")
                    break
                cnt += 1
            if res[idx] == 0:
                outl[3] = 'NONE'
            elif res[idx] == 1:
                outl[3] = 'AGAINST'
            elif res[idx] == 2:
                outl[3] = 'FAVOR'
            outf.write("\n" + "\t".join(outl))

    outf.close()



def eval(file_gold, file_pred, evalscript="../eval.pl"):
    """
    Evaluate using the original script, needs to be in same format as train/dev data
    :param file_gold: testing file with gold standard data
    :param file_pred: file containing predictions
    :param evalscript: file location for official eval script
    """
    pipe = subprocess.Popen(["perl", evalscript, file_gold, file_pred], stdout=sys.stdout)
    pipe.communicate()