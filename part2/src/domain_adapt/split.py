""" Split questions from src and tgt domain into train and test set. For Domain Classifier """

import numpy as np
import os 
import sys
import torch

from tqdm import tqdm
def split_data(data_file, train_file, test_file, ratio=0.8):
	f = open(data_file)
	lines = f.readlines()
	
	split_at = int(len(lines) * ratio)
	train_lines = lines[:split_at]
	test_lines = lines[split_at:]

	def out(out_file, out_lines):
		f = open(out_file, "w")
		for line in tqdm(out_lines, desc="Writing"):
			f.write(line)
		f.close()

	out(train_file, train_lines)
	out(test_file, test_lines)

if __name__ == "__main__":
	src_file = "../../data/QR/text_tokenized.txt"
	tgt_file = "../../data/Android/corpus.tsv"
	src_train = "../../data/domain_classifier/src.train"
	src_test = "../../data/domain_classifier/src.test"
	tgt_train = "../../data/domain_classifier/tgt.train"
	tgt_test = "../../data/domain_classifier/tgt.test"

	split_data(src_file, src_train, src_test)
	print "==> Finish Splitting Source Domain Data"	
	split_data(tgt_file, tgt_train, tgt_test)
	print "==> Finish Splitting Target Domain Data"	
