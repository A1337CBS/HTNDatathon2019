# HTNDatathon2019
Hack the news Datathon, Jupyter notebooks we made for the event


This article describes our submission for the Hack the News Datathon 2019 which focuses on Task 2, Propaganda sentence classification. It outlines our exploratory data analysis, methodology and future work. Our work revolves around the BERT model as we believe it offers an excellent language model that’s also good at attending to context which is an important aspect of propaganda detection.

Members(name, nickname and email):
Ahmed(ahmed, ahmedaziz.nust@hotmail.com), Shehel(shehel, shehel@live.com) and Faher(blackhoodie)
Business Understanding

Propaganda has become prevalent in news articles nowadays which damages the credibility of news outlets. Every day hundreds of new articles are created and there is no way to check their credibility. Many solutions have been tried to mitigate this, recent methods using machine learning have become quite popular and prevalent.

Supercharge Propaganda with the internet and ad revenue model and you get fake news and a distorted reality. Its a self perpetuating cycle. But there are patterns in propagandist language one can see and this can be learned using ML.  Despite the classification being subjective sometimes, this is a valid hypothesis as shown by the excellent results we were able to obtain in the dataset.

Detecting propagandist patterns is very dependent on the context and building word embeddings that takes into account the context where it’s used and this is why we decided to use the recent BERT architecture by Google.

We attempted task 2 which is to classify sentences as containing propaganda or not. We chose this as the first task because we found that it simplifies the other two tasks. Task 1 can be trivially done with a good model in Task 2 by using the model to classify each sentence in the article and using this to classify the article itself. The same model can be used with Task 3 where transfer learning can be used.
Data Understanding

Data Imbalance

72% of the sentences in the 15170 sentences long training set for Task 2 was non-propaganda and 28% was propaganda . We hope to correct for this in the future using under sampling and oversampling.

Sentence length 5 Number Summary
	Min 	Q1 	Median 	Q3 	Max
Non-Propaganda 	1 	12 	21 	33 	150
Propaganda 	2 	18 	29 	44 	177

We decided to go for Sequence Length of 120 for our models. Another interesting observation is that Propaganda on average contains more words in sentences which is understandable as propaganda is simply exaggeration or added false data containing little information.

Furthermore, we also found hat nouns are more common in sentences containing propaganda which is intuitive since propaganda usually involves invoking some named entity such as a geographic location, individual and so on. This led us to use cased tokenizations as we thought this information is important in task 2 and more so in task 3.
Data Preparation

    Firstly, we drop all empty lines in the dataset and automatically label it as non-propaganda.
    Word-piece cased tokenization

 
Modeling

Our model revolves around the very popular and current state of the art BERT architecture and the reasons can be summed by the three points below:

    Pre-training a neural network model on a known task and then performing fine-tuning has repeatedly shown that it saves time, gives access to better representations and is intuitively a better way to train a model
    Applying it to several well know tasks, BERT has shown that it learns an excellent representation of the language
    A major reason behind this is how well it learns the embeddings by taking into account the context. It corrects directionality problems of previous architectures and considers the context in totality.

Our initial goal was to fine-tune the model for the task. However, we found that we lacked computation resources to train an appropriate model given the time restrictions. And the model we were able to train was unstable and gave us poor results. Thus, we decided to use pre-trained BERT embeddings and then use them to learn a good classifier. The hope is that the context encoded in BERT embeddings can be utilized to learn distinguishing patterns that appear in propagandist statements.

Our base-line utilized Naive Bayes with BERT embeddings. We also tried several other tools including SVM, Adaboost, Random Forest and KNN which all gave scores close to the baseline. Finally, we found that we could improve on the baseline by using neural networks.

Neural networks were quite sensitive to certain settings and we had to work out what works best by trial and error. We started with multi-layer perceptrons with a single intermediate layer and experimented with different architectures.

We tried several techniques to improve the BERT embeddings+MLP model. An ensemble of 4 MLP with varying architecture and a Naive Bayes model gave us our best F1 score of 0.59. We also tried pseduo-labelling whereby we created predictions on the dev set and added this to our training set, creating a larger train set. Another method was to average the output of previous submissions. However, this didn’t return significant gains, perhaps due to weak underlying models.

We also tried fine-tuning ELMo embeddings but these were too slow and not as good as BERT embeddings.
Model 	F1
Fine-tuned BERT base [20 epochs] (uncased) 	.39 (slow but increasing)
BERT Embeddings with Naive Bayes 	.55
BERT Embeddings with Adaboost 	.56
Fine tuned ELMo embeddings 	.56
BERT Embeddings with MLP 	.57
BERT Embeddings Ensemble 4 MLP Naive Bayes 	.59
Baseline (provided by DSS) 	0.42
Evaluation

F1 score, precision, accuracy and recall were our primary metrics.

 
Future Work

Task 2

Successfully fine-tune BERT for sentence level classification. A good model for sentence classification will also simplify Task 2.

Task 1

Use Task 2 model for Task 1. If one or a couple of sentences is classified as propaganda in an article, the article is classified a propaganda. This will be a more strict classifier.

Task 3

We made good progress in building the model for Task 3 which we thought of framing it like a Named Entity Recognition problem. But found that converting span information into labels can be very time consuming since spans cover more than a sentence in more than a few cases and our model was built to work at the sentence level. However, this is doable with careful pre-processing and post-processing.

For example a sentence “The President is a liar” will be tokenized into [The, President, is, a, liar] and will have labels as [0, 0, 0, 0, 1] where 0 corresponds to ordinary word and 1 corresponds to a propaganda technique (name-calling). This information needs to be derived from the span and this is a problem we can indeed solve to use our existing BERT model.
Resources

    16 GB RAM, 1080 Ti GPU
    PyTorch, Scikit-learn, Keras, Tensorflow, Pandas
    https://github.com/huggingface/pytorch-pretrained-BERT
    https://github.com/hanxiao/bert-as-service

Code

The below github links contains our code, the first link is for the model that we used for our final test and dev results as shown on leaderboard. The second model is the one we believed we could get best results from but were not able to fine tune it. The third link provides the Python code we used to create the dataset for Task 2.

https://github.com/A1337CBS/HTNDatathon2019/blob/master/BERT_as_service(1).ipynb
https://github.com/A1337CBS/HTNDatathon2019/blob/master/BERT_task2.ipynb
https://github.com/A1337CBS/HTNDatathon2019/blob/master/CreateTask2Dataset.py
