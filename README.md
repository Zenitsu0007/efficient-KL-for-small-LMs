# Setup

```bash
!pip install pytorch_lightning
!pip install transformers
!pip install datasets
!pip install nltk
!pip install rouge_score
```

# Download datasets

```bash
!mkdir data
!wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz -P data
!tar -xvf data/paws_wiki_labeled_final.tar.gz -C data
!mv data/final/* data
!rm -r data/final
!rm -r data/paws_wiki_labeled_final.tar.gz
```

# Run

```bash
python T5/T5_finetune.py
```
