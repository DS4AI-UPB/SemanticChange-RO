# Semantic Change Detection for the Romanian Language

## Article:

Ciprian-Octavian Truică, Victor Tudose, Elena-Simona Apostol. *Semantic Change Detection for the Romanian Language*. Arxiv, 2023. Link: [https://arxiv.org/TBA](https://arxiv.org/TBA)

## Code 

Packages needed:
- scipy
- Scikit-learn
- numpy
- nltk
- gensim
- matplotlib
- pandas
- flask
- plotly


### Static word embeddings

Use the classes from the ``sgns_op.py`` and ``sgns_wi.py`` files located in the ``representations`` folder to train word embeddings.

### Contextual word embeddings

To train and test the ELMo embeddings use the jupyter notebooks in the ``elmo_embs`` folder.
You will need the AllenNLP package.

### Running the tests

A test is comprises by runing a model with a predefined configuration: 

SGNS-OP: ``run.py sgns_op tasks/sem_shift_en.json``

SGNS-WI: ``run.py sgns_wi tasks/sem_shift_en.json``

ELMO-PREV: ``run.py elmo_with_precomp tasks/elmo_model1.json``

ELMO-POST: ``run.py elmo_with_precomp tasks/elmo_model2.json``


Schema for a configration task.json file (in the tasks directory):

	target: where to output data
	language, name, description: descriptive terms to describe the task
	corpora: a list of 2 corpora used to compare the words
	threshold: a value above which we consider the word to be changed
	tests: a list of words to be compared across corpora, specifing the word and expected change
	skip_train: a value that if it present, will make the test skip any train and just compare results


### Demo

Change directory to the ``demo`` folder and run ``python app.py``
Use  a browser to and open the link [http://127.0.0.1:5000](http://127.0.0.1:5000)
