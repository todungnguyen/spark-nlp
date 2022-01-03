Pretrained Model doesnt work with test set 

Solution 1
https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public

- [x] 1 count result by ner_chunk, not ner by word (compare v1 - v2 in zeppelin)

- [ ] 2 Pre-processing 
        - [x] replace null, 0
        - [ ] spell check ex: allee => allée (QUE ANGLAIS AVEC SPARK NLP)
			=> créer notre french model 

	BLOCKEE - java.lang.UnsatisfiedLinkError: no tensorflow_cc in java.library.path

	https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc
	https://nlp.johnsnowlabs.com/docs/en/annotators#contextspellchecker
	https://towardsdatascience.com/training-a-contextual-spell-checker-for-italian-language-66dda528e4bf

- [x] 3 Check each line to find what wrong (ex: detect RUE as LOC but not AVENUE) then try to create a train set like test set => AUCUN REGLE ????
	allee + personne => PER
	allee + de => LOC (parfois MISC)
	allée pareil
	pas détecter avenue, av
	chemin 50%

- [ ] 4 Test with different step in model

Presentation
https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/slides/Spark%20NLP%20Training%20-%20Public%20-%20Jan%202021.pdf

https://nlp.johnsnowlabs.com/docs/en/pipelines#french-entityrecognizermd

List pretrained pipeline
https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb

Custom model
https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/4.NERDL_Training.ipynb
https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/3.NER_with_BERT.ipynb


        - [ ] Word Embedding: Glove, Bert, xling, xlm-robert
        - [ ] Word Embedding  Compare + Pipeline steps in spark NLP
		https://towardsdatascience.com/glove-elmo-bert-9dbbc9226934
        - [ ] nerdlmodel	

￼


some pretrained models require specific types of embeddings, depending on which they were trained on. For example, the default model "ner_dl" requires the WordEmbeddings "glove_100d"

corpus, embedding, model


Solution 2 - camemBERT (meilleur accuracy, mais très lent)

camemBERT train a model require large data and powerful machine, robert architect
https://huggingface.co/camembert-base
https://medium.com/@vitalshchutski/french-nlp-entamez-le-camembert-avec-les-librairies-fast-bert-et-transformers-14e65f84c148

Different NLP models in French
https://piaf.etalab.studio/francophonie-ia-english/

Fine-tuning = the process in which parameters of a model must be adjusted very precisely in order to fit with certain observations

How to fine tuning a pretrained model
https://huggingface.co/docs/transformers/training

Transformer = a library which provides many pretrained models (camemBERT inclus)
https://huggingface.co/docs/transformers/index

State-of-art = the best model
https://www.quora.com/What-does-‘state-of-the-art’-mean-in-machine-learning-Is-it-the-best-or-one-of-the-best

Comment créer un model avec camemBERT
https://www.kaggle.com/houssemayed/camembert-for-french-tweets-classification#Defining-the-parameters-and-metrics-to-optimize

https://ledatascientist.com/analyse-de-sentiments-avec-camembert/


camemBERT-ner

- [x] fine-tuned model => work well with train set => test on bourso set
https://huggingface.co/Jean-Baptiste/camembert-ner

- [x] can’t load model on bourso pc (problem of ssl) => download model to local on mac => github => bourso => hdfs
        - [x] git login: todungnguyen + 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


Solution 3 - flauBERT -  que pre-training BERT (embedding, il faut faire model après)

https://github.com/getalp/Flaubert

How to use BERT for the first time
https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/















