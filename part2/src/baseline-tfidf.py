import torch
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc

from tqdm import tqdm
from meter import AUCMeter

if __name__ == "__main__":
	datadir = "../data/Android/"

	# read all questions
	f = open(datadir + "corpus.tsv")
	questions = []
	id2ind = {}   # qid to index in questions
	ind = 0
	for line in f:
		parts = line.split("\t")
		assert len(parts) == 3
		id2ind[int(parts[0])] = ind
		questions.append(parts[1] + " " + parts[2])
		ind += 1
	f.close()

	# get tfIdf vectorizer
	vectorizer = TfidfVectorizer()  # TODO(demi): change the hyperparameters
	weight_csr_matrix = vectorizer.fit_transform(questions)


	datafiles = ["dev", "test"]

	for datafile in datafiles:
		print "=> Running on Data %s" % datafile

		y_true = []
		y_score = []
		meter = AUCMeter()
		for label, suffix in enumerate([".neg.txt", ".pos.txt"]):
			print "===> Running on suffic %s" % suffix

			f = open(datadir + datafile + suffix)
			lines = f.readlines()
			for i in tqdm(range(len(lines)), desc="Evaluating"):
				line = lines[i]
				q1, q2 = line.split(" ")
				q1, q2 = int(q1), int(q2)

				assert q1 in id2ind and q2 in id2ind
				q1_tensor = torch.from_numpy(weight_csr_matrix[id2ind[q1]].toarray()).float()
				q2_tensor = torch.from_numpy(weight_csr_matrix[id2ind[q2]].toarray()).float()

				# print out size
				score = torch.nn.CosineSimilarity()(q1_tensor, q2_tensor)
				#print "label=", label, "score=", score[0]
				y_true.append(label)
				y_score.append(score[0])
			f.close()

		meter.add(np.array(y_score), np.array(y_true))
		auc_005 = meter.value(0.05)
		print "auc_005=", auc_005
