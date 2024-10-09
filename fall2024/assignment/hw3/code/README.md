## Set up environment
We recommend you to set up a conda environment for packages used in this homework.
```
conda create -n hw3-nlp python=3.9
conda activate hw3-nlp
pip install -r requirements.txt
```

After this, you will need to install certain packages in nltk
```
python3
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download(’punkt’)
>>> exit()
```
