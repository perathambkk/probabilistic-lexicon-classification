# A Python3 compatible fork of Unsupervised Learning for Lexicon-Based Classification

This is the implementation from the eponymous AAAI 2017 paper. 

Please cite
```
Eisenstein, Jacob. "Unsupervised learning for lexicon-based classification." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 31. No. 1. 2017.
```

BibTex:
```
{@inproceedings{eisenstein2017unsupervised,
  title={Unsupervised learning for lexicon-based classification},
  author={Eisenstein, Jacob},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={31},
  number={1},
  year={2017}
}
```
------------------------------------------------------------------------------------------------------------
It includes:

- My code
- The Bing Liu lexicon (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)
- The Cornell sentiment polarity dataset, version 2.0 (https://www.cs.cornell.edu/people/pabo/movie-review-data/)

I cannot promise to provide much support for the code, but I will try.

# Reproducing results

To reproduce the Cornell dataset results, you can run the following line:

```python bayeslex.py --epochs 250 cornell liu-pos.utf8 liu-neg.utf8 --optimizer admm --prefilter```

The second column of the output contains the AUC numbers reported in the paper.

You can get the other datasets in the paper at the following locations:

- Stanford/IMDB: http://ai.stanford.edu/~amaas/data/sentiment/
- Amazon: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
- CorpusCine: http://www.lsi.us.es/~fermin/index.php/Datasets

The ISOL leixcon was obtained here: http://metashare.upf.edu/repository/browse/isol/f93c81ba65c911e48763000c291ecfc893e66d98695145c0b0066cfa00e6ccda/
