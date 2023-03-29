# Natural-Language-Processing-NLP-
This repo contains various NLP projects, such as NER, Semantic Role Labelling and Conference resolution

## Semantic Role Labelling

Semantic Role Labeling (SRL) is a natural language processing (NLP) task that involves identifying the semantic relationships between the arguments and predicates in a sentence. The aim of SRL is to understand the meaning of a sentence, and it is a crucial component in many NLP applications.

In this project, we perform Predicate Disambiguation, Argument Identification, and Argument Classification on an English dataset using XLM-RoBERTa, a multilingual transformer-based language model that generates contextualized word embeddings.We also explore how to fine-tune the model for lower resource languages (French and Spanish for our dataset), using a multi-task learning approach. 

In the implementation of SRL, XLM-RoBERTa, a transformer-based language model pre-trained on a large multilingual corpus, is used to encode the input text and generate word embeddings. The training process involves training the models on a primary dataset in English, which is divided into two tasks:

- Task 1: given predicate location and predicate sense, the model identifies and classifies the arguments; 
* Task 2:  given only the predicate location, the model does predicate disambiguation, argument classification, and argument identification.

To fine-tune the models for secondary languages with lower annotated data, a multi-task learning approach is used with two strategies. In strategy 1, the trained model is used as the base model, and a new layer is added after removing the last layer. The weights of the base model are frozen, and the model is trained on the secondary language.

In strategy 2, the base model is used as it is, and the weights of all layers except the last one are frozen. The performance of both strategies is compared, and it is found that the performance of strategy 1 is better for task 2, while strategy 2 performs well for task 1. This approach demonstrates how the knowledge acquired by a model on one language can be transferred to another language with less annotated data.

**Check out the report in this repo to know more in details**

## Conference resolution

This project is focused on solving the problem of conference resolution in natural language processing (NLP). Conference resolution involves identifying entities mentioned in a text and linking them to their corresponding mentions. This is a crucial step in understanding the text and extracting relevant information from it. In this project, the focus is specifically on the third step of the conference resolution pipeline, which is predicting the correct entity that an ambiguous pronoun refers to.

To solve this problem, the project used a combination of BERT embeddings and a Bi-LSTM architecture. BERT is a pre-trained deep learning model that is capable of generating contextualized word embeddings. The Bi-LSTM architecture consists of two layers that can capture the context and dependencies between words in the input text.

During the training process, the output of the Bi-LSTM for the ambiguous pronoun and the candidate entities were fed into a multi-layer perceptron (MLP) to make predictions. Two approaches were tested for feeding the information into the MLP, and both produced similar results.The first approach involved directly feeding the output from the Bi-LSTM for the pronoun and the Bi-LSTM for the two candidate entities into the MLP. The second approach involved subtracting the output of one candidate entity (A) from the output of the pronoun and dividing the result by two, and doing the same for the other candidate entity (B), before feeding the results into the MLP.

The project also experimented with using SpanBERT embeddings instead of BERT embeddings. SpanBERT is a variation of BERT that is designed to better represent and predict spans of text. However, the results from all experiments were mostly comparable, with BERT embeddings performing slightly better and faster to converge.

Overall, the project demonstrates the effectiveness of using pre-trained language models like BERT in solving NLP tasks like conference resolution. The combination of BERT embeddings and a Bi-LSTM architecture proved to be effective in predicting the correct entity that an ambiguous pronoun refers to. Future work may involve exploring other pre-trained models or fine-tuning strategies to further improve the performance of the model.

**Check out the report in this repo to know more in details**



## NER (Named Entity Recognition)

This project is an implementation of Named Entity Recognition (NER) using a combination of deep learning techniques such as LSTM and Bi-LSTM architectures with pre-trained word embeddings from GloVe. The aim of this project is to classify named entities in a given text or sentence into different categories such as groups, persons, locations, corporations, products, creative works, etc. The project uses BIO tags to identify and label the different named entities. The models were trained on a highly unbalanced dataset, and the performance of the models was evaluated using f1-score and precision. The project also explores the impact of pre-trained embeddings and different architectures on the performance of the models. The results of the project showed that using a Bi-LSTM architecture with pre-trained GloVe embeddings yielded the best precision score. This project provides insights and code for implementing NER models in natural language processing applications.

**Check out the report in this repo to know more in details**

