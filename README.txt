Hierarchical Embedding Model for Personalized Product Search

Overview

  This is an implementation of the Hierarchical Embedding Model (HEM) for personalized product search [2].
   Please cite the following paper if you plan to use it for your project：
    
    Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, W. Bruce Croft. 2017.
      Learning a Hierarchical Embedding Model for Personalized ProductSearch.
      In Proceedings of SIGIR ’17
    	
  The HEM is a deep neural network model that jointly learn latent representations for queries, products 
  and users.  It is designed as a generative model and the embedding representations for queries, users 
  and items in the HEM are learned through optimizing the log likelihood of observed user-query-item 
  purchases.  The probability (which is also the rank score) of an item being purchased by a user with
  a query can be computed with their corresponding latent representations.  Please refer to the paper 
  for further details.


Requirements

  1. To run the HEM model in ./HEM/ and the python scripts in ./scripts/, python 2.7+ and 
     Tensorflow v1.0+ are needed.

  2. To run the jar package in ./jar/, JDK 1.7 is needed.

  3. To compile the java code in ./java/, Galago from Lemur Project 
     (https://sourceforge.net/p/lemur/wiki/Galago%20Installation/) is needed. 


Data Preparation

  1. Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/ (e.g. In our paper, we 
     used 5-core data).

  2. Stem and remove stop words from the Amazon review datasets if needed (e.g. In our paper, we stem 
     the field of “reviewText” and “summary” without stop words removal).

     java -Xmx4g -jar ./jar/AmazonReviewData_preprocess.jar <jsonConfigFile> <review_file> <output_review_file>

       where
       <jsonConfigFile>       A JSON file that specifies the file path of stop words list.  An example 
                                can be found in the root directory.  Enter “false” if you don’t want to 
                                remove stop words. 
       <review_file>          The path for the original Amazon review data.
       <output_review_file>   The output path for processed Amazon review data.

  3. Index datasets

     python ./scripts/index_and_filter_review_file.py <review_file> <indexed_data_dir> <min_count>

       where

       <review_file>          The file path for the Amazon review data.
       <indexed_data_dir>     Output directory for indexed data.
       <min_count>            The minimum count for terms.  If a term appears less then <min_count>
                                times in the data, it will be ignored.

  4. Extract queries and Split train/test

     o Download the meta data from http://jmcauley.ucsd.edu/data/amazon/ 

     o Match the meta data with the indexed data:
         java -Xmx16G -jar ./jar/AmazonMetaData_matching.jar <jsonConfigFile> <meta_data_file> <indexed_data_dir>

         where
         <jsonConfigFile>     A JSON file that specifies the file path of stop words list.  An example can be 
                                found in the root directory.  Enter “false” if you don’t want to remove stop words. 
         <meta_data_file>     The path for the meta data.
         <indexed_data_dir>   The directory for indexed data.

     o Split datasets for training and test
         python ./scripts/split_train_test_data.py <indexed_data_dir> <review_sample_rate> <query_sample_rate>

         where
         <indexed_data_dir>    The directory for indexed data.
         <review_sample_rate>  The proportion of reviews used in test for each user (e.g. in our paper, we used 0.3).
         <query_sample_rate>   The proportion of queries used in test (e.g. in our paper, we used 0.3).


Model Training/Testing
  python ./HEM/main.py --<parameter_name> <parameter_value> --<parameter_name> <parameter_value> … 

  where parameter names and values are defined as follows:

    o learning_rate                The learning rate in training.  Default 0.05.

    o learning_rate_decay_factor   Learning rate decays by this much whenever the loss is higher than three 
                                   previous loss.  Default 0.90.

    o max_gradient_norm            Clip gradients to this norm. Default 5.0

    o subsampling_rate             The rate to subsampling. Default 1e-4. 

    o L2_lambda                    The lambda for L2 regularization. Default 0.0

    o query_weight                 The weight for queries in the joint model of queries and users. Default 0.5

    o batch_size                   Batch size used in training.  Default 64.

    o data_dir                     Data directory, which should be the <indexed_data_dir>.

    o input_train_dir              The directory of training and testing data, which usually is <data_dir>/query_split/.

    o train_dir                    Model directory & output directory.

    o similarity_func              The function to compute the ranking score for an item with the joint model of 
                                   query and user embeddings.  Default “product”.
      -- “product”                   The dot product of two vectors.
      -- “cosine”                    The cosine similarity of two vectors.
      -- “bias_product”              The dot product plus a item-specific bias.

    o net_struct                   Network structure parameters.  Different parameters are separated by “_” (e.g.).
                                   Default “simplified_fs”.
      -- “LSE”                       The latent space entity model proposed by Gysel et al. [1]
      -- “simplified”                Simplified embedding-based language models without modeling for each review. [2]
      -- “pv”                        Embedding-based language models with review modeling using paragraph vector model. [3]
      -- “hdc”                       Regularized embedding-based language models with word context. [4]
      -- “mean”                      Average word embeddings for query embeddings. [5]
      -- “fs”                        Average word embeddings with non-linear projection for query embeddings. [1]
      -- “RNN”                       Recurrent neural network encoder for query embeddings.

    o embed_size                   Size of each embedding.  Default 100.

    o window_size                  Size of context window for hdc model.  Default 5.

    o max_train_epoch              Limit on the epochs of training (0: no limit).  Default 5.

    o steps_per_checkpoint         How many training steps to do per checkpoint.  Default 200

    o seconds_per_checkpoint       How many seconds to wait before storing embeddings.  Default 3600

    o negative_sample              How many samples to generate for negative sampling.  Default 5.

    o decode                       Set to “False" for training and “True" for testing.  Default “False"

    o test_mod                     Test modes.  Default “product_scores".
      -- “product_scores”            Output ranking results and ranking scores.
      -- “output_embedding"          Output embedding representations for users, items and words.

    o rank_cutoff                  Rank cutoff for output rank lists.  Default 100.


Evaluation

  o After training with "--decode False”, generate test rank lists with "--decode True”.

  o TREC format rank lists for test data will be stored in <train_dir> with name “test.<similarity_func>.ranklist”

  o Evaluate test rank lists with ground truth <input_train_dir>/test.qrels using trec_eval or galago eval tool.


References

  [1] Christophe Van Gysel, Maarten de Rijke, and Evangelos Kanoulas. 2016. Learning latent vector
      spaces for product search. In Proceedings of the 25th ACM CIKM.

  [2] Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, W. Bruce Croft. 2017. Learning a Hierarchical
      Embedding Model for Personalized ProductSearch. In Proceedings of SIGIR ’17.

  [3] Quoc V Le and Tomas Mikolov. 2014. Distributed Representations of Sentencesand Documents.
      In ICML.

  [4] Sun, Fei, Jiafeng Guo, Yanyan Lan, Jun Xu, and Xueqi Cheng. 2015 "Learning Word Representations
      by Jointly Modeling Syntagmatic and Paradigmatic Relations."  In ACL.

  [5] Ivan Vulic ́ and Marie-Francine Moens. 2015. Monolingual and cross-lingual in-formation retrieval
      models based on (bilingual) word embeddings.  In Proceedingsof the 38th ACM SIGIR.
