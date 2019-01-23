import numpy as np
import json
import random
import gzip
import math

class Tensorflow_data:
	# data_path 是index的路径
	# input_train_dir是split的路径
	# set_name表明是train或者test的数据
	def __init__(self, data_path, input_train_dir, set_name):
		#get product/user/vocabulary information
		self.product_ids = [] # review信息中得到的所有product（被购买过）
		with gzip.open(data_path + 'product.txt.gz', 'r') as fin:
			for line in fin:
				self.product_ids.append(line.strip())
		self.product_size = len(self.product_ids)
		self.user_ids = [] # review信息中得到的所有user （在本数据中出现的都是买了5次以上物品的用户）
		with gzip.open(data_path + 'users.txt.gz', 'r') as fin:
			for line in fin:
				self.user_ids.append(line.strip())
		self.user_size = len(self.user_ids)
		self.words = [] # review信息中得到的所有word
		with gzip.open(data_path + 'vocab.txt.gz', 'r') as fin:
			for line in fin:
				self.words.append(line.strip())
		self.vocab_size = len(self.words)
		self.query_words = [] # 每一行是一个query
		self.query_max_length = 0
		with gzip.open(input_train_dir + 'query.txt.gz', 'r') as fin:
			for line in fin:
				line = bytes.decode(line)
				words = [int(i) for i in line.strip().split(' ')]
				if len(words) > self.query_max_length:
					self.query_max_length = len(words)
				self.query_words.append(words)
		#pad
		# 补全query为最大长度（在头部填充-1）
		for i in range(len(self.query_words)):
			self.query_words[i] = [-1 for j in range(self.query_max_length-len(self.query_words[i]))] + self.query_words[i]


		#get review sets
		self.word_count = 0
		# 计算每条评论的出现次数
		self.vocab_distribute = np.zeros(self.vocab_size) 
		# 得到用户购买的物品对(user_idx, product_idx)
		self.review_info = []
		# 
		self.review_text = []
		with gzip.open(input_train_dir + set_name + '.txt.gz', 'r') as fin:
			for line in fin:
				line = bytes.decode(line)
				arr = line.strip().split('\t')
				self.review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
				self.review_text.append([int(i) for i in arr[2].split(' ')])
				# 对新加入的词计数
				for idx in self.review_text[-1]:
					self.vocab_distribute[idx] += 1
				self.word_count += len(self.review_text[-1])
		# 一共有多少review
		self.review_size = len(self.review_info)
		self.vocab_distribute = self.vocab_distribute.tolist() 
		self.sub_sampling_rate = None
		# 
		self.review_distribute = np.ones(self.review_size).tolist()
		self.product_distribute = np.ones(self.product_size).tolist()

		#get product query sets
		self.product_query_idx = []
		with gzip.open(input_train_dir + set_name + '_query_idx.txt.gz', 'r') as fin:
			for line in fin:
				line = bytes.decode(line)
				arr = line.strip().split(' ')
				query_idx = []
				for idx in arr:
					if len(idx) < 1:
						continue
					query_idx.append(int(idx))
				self.product_query_idx.append(query_idx)

		print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size, 
					self.review_size, self.user_size, self.product_size))

	def sub_sampling(self, subsample_threshold):
		if subsample_threshold == 0.0:
			return
		self.sub_sampling_rate = np.ones(self.vocab_size)
		threshold = sum(self.vocab_distribute) * subsample_threshold
		count_sub_sample = 0
		for i in range(self.vocab_size):
			#vocab_distribute[i] could be zero if the word does not appear in the training set
			self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]+1e-5),
											1.0)
			count_sub_sample += 1

	def read_train_product_ids(self, data_path):
		self.user_train_product_set_list = [set() for i in range(self.user_size)]
		self.train_review_size = 0
		with gzip.open(data_path + 'train.txt.gz', 'r') as fin:
			for line in fin:
				line = bytes.decode(line)
				self.train_review_size += 1
				arr = line.strip().split('\t')
				self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))


	def compute_test_product_ranklist(self, u_idx, original_scores, sorted_product_idxs, rank_cutoff):
		product_rank_list = []
		product_rank_scores = []
		rank = 0
		for product_idx in sorted_product_idxs:
			if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]):
				continue
			product_rank_list.append(product_idx)
			product_rank_scores.append(original_scores[product_idx])
			rank += 1
			if rank == rank_cutoff:
				break
		return product_rank_list, product_rank_scores

	def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
		with open(output_path + 'test.'+similarity_func+'.ranklist', 'w') as rank_fout:
			for uq_pair in user_ranklist_map:
				user_id = self.user_ids[uq_pair[0]]
				for i in range(len(user_ranklist_map[uq_pair])):
					product_id = self.product_ids[user_ranklist_map[uq_pair][i]]
					rank_fout.write(user_id+'_'+str(uq_pair[1]) + ' Q0 ' + product_id + ' ' + str(i+1)
							+ ' ' + str(user_ranklist_score_map[uq_pair][i]) + ' ProductSearchEmbedding\n')



	def output_embedding(self, embeddings, output_file_name):
		with open(output_file_name,'w') as emb_fout:
			try:
				length = len(embeddings)
				if length < 1:
					return
				dimensions = len(embeddings[0])
				emb_fout.write(str(length) + '\n')
				emb_fout.write(str(dimensions) + '\n')
				for i in range(length):
					for j in range(dimensions):
						emb_fout.write(str(embeddings[i][j]) + ' ')
					emb_fout.write('\n')
			except:
				emb_fout.write(str(embeddings) + ' ')





