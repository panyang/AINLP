IMDB Movie Review Data Sentiment Analysis from Scratch
=========================================
Data
---
Get the original data from [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
Download link: [Large Movie Review Dataset v1.0](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

How to
---
* virtualenv venv & source venv/bin/activate & pip install -r requirement.txt
* python build_word_index.py to get json word index
* python build_data_index.py to get train and test npz data
* Use aclimdb.py to support local data made by two above steps
* Refer more from my Chinese blog：http://www.52nlp.cn/?p=10537
