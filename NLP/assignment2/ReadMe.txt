			                            Report for Assignment 2


Varun Kumar S   01FB15ECS338
Varun V         01FB15ECS341
Varun Y Vora    01FB15ECS342
Vishwas Sathish  01FB15ECS355


Summary :-

1. INPUT

->shape of input vector (20, 100)
->each word is represented as a vector of 100 features
->sentences with more than 20 words are clipped
->and one with less than 20 words are padded with zero vectors
->target : one hot vectors of dimension = length of vocabulary

2. OUTPUT
->softmax output 

Approach Taken :-

1. First we load all the tweet data from the file 'consolidate.csv' and separately store original and corrected 
tweets in two lists after tokenizing them. We used nltk library to tokenize sentences into a list of words.

2. The original data is then preprocessed. Each word is converted to its lowercase words, to maintain uniformity 
in the dataset. The corrected tweets are processed to find all the unique words and their count. Only a subset of 
these unique words are chosen according to their occurrence count for our bag of words.

3. We Created one-hot vectors for each word in our bag of words, which will constitute our 'expected output' data.
Each word in the original tweet dataset is converted to its corresponding vector. Gensim's word2vec was used for 
this purpose.

4. Now that we had all the required data in their proper format, we segregated and randomly chose X(input) and 
y(expected output) vectors from the dataset. This data was split into training(4050) and validation data(50). 

5. For our network model, we used encoder-decoder RNN model using keras. Our Training set from X is sent as the input
and trained for the expected output set from y. Categorical cross entropy was used as our error function and 
softmax was used for activation.

Result :-

Validation Accuracy : 38.54 % for 10 epochs

Possible reasons for low accuracy : 

1. Low quality of the dataset used for training. Since manual correction for every tweet was done, human errors would have 
hindered the quality of correction and possibly reduced the accuracy.

2. Small size for bag of words. We used only 3500 most occurinng words for the given dataset, as the size of our one-hot vector 
grows with the size of bag of words. This might not have been enough to predict corrected outputs as there were over 15,000 
unique words.

3. Smaller size of the dataset used. Less than 5000 tweets were used for training purpose.

4. Number of epochs and other Model parameters like error function used, activation function and optimizer. 


