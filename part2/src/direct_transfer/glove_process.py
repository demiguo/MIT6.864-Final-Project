# TODO(demi): glove processing file that retrieve glove vectors related to all files in corpora
import numpy as np
import nltk
from tqdm import tqdm
from sets import Set
# first vocab = <EMPTY>, second vocab = <UNK>

my_files = ["../../data/Android/corpus.tsv", "../../data/QR/text_tokenized.txt"]

def run(files=my_files):
	embedding_dim = 300

	w2i = {}
	i2w = {}

	f_e = open("../../data/vector/glove.vectors", "w")
	f_v = open("../../data/vector/glove.vocab", "w")
	w2i["<EMPTY>"] = 0
	i2w[0] = "<EMPTY>"

	w2i["<UNK>"] = 1
	i2w[1] = "<UNK>"

	vocab_size = 2

	# first construct vocab
	vocab = Set([])

	for file in files:
		f = open(file)
		lines = f.readlines()
		for line in tqdm(lines, "Vocab"):
			idx, title, body = line.split("\t")

			title = title.split(" ")
			body = body.split(" ")
			for word in title:
				vocab.add(word.replace("\n",""))
			for word in body:
				vocab.add(word.replace("\n",""))
		f.close()
	print "=> Finish Vocabulary Construction"

	f_v.write("<EMPTY>")
	f_v.write("<UNK>")
	f_e.write(" ".join(["0"] * embedding_dim) + "\n")
	f_e.write(" ".join(["0"] * embedding_dim) + "\n")

	f_g = open("../../data/vector/glove.840B.300d.txt")
	lines = f_g.readlines()
	for line in tqdm(lines, desc="Vector"):
		parts = line.split(" ")  # NB(demi): with "\n"
		word = parts[0]
		vec_str = " ".join(parts[1:])
		#print "word", word
		#print "vec_str=", vec_str, "[END}"
		
		if (word in vocab) and (not word in w2i):
			# keep
			w2i[word] = vocab_size
			i2w[vocab_size] = word
			f_e.write(vec_str)
			f_v.write(word + "\n")
			vocab_size += 1

	f_g.close()

	f_e.close()
	f_v.close()

if __name__ == "__main__":
	run()