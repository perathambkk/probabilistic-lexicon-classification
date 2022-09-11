'''
This collects all the data munging code.

See data.ipynb for usage
'''

from glob import glob
from collections import Counter
from nltk.tokenize import word_tokenize,sent_tokenize
import codecs
import xml.etree.ElementTree as ET
import os


#http://ai.stanford.edu/~amaas/data/sentiment/
imdb_base = 'aclImdb/test/'

#http://www.lsi.us.es/~fermin/index.php/Datasets
cine_base = 'corpusCriticasCine.fixed'

#https://www.cs.cornell.edu/people/pabo/movie-review-data/
cornell_base = 'txt_sentoken'

#https://www.cs.jhu.edu/~mdredze/datasets/sentiment/
amazon_base = 'amazon-original-from-dredze'

def counterString(counts):
    return ' '.join(['%s:%s'%(key,val) for key,val in counts.items()])

def getIMDBWordCounts(docname):
    counter = Counter()
    with codecs.open(docname,mode='r',encoding='utf-8') as fin:
        for line in fin:
            for sent in sent_tokenize(line):
                counter += Counter([x.lower() for x in word_tokenize(sent) if x.isalpha()])
    return counterString(counter)

def getCorpusCineCountsAndLabels(filename):
    parser = ET.XMLParser()
    with open(filename,mode='r') as fin:
        tree = ET.fromstring(fin.read(),parser=parser)
        rank = int(tree.find('.').attrib['rank'])
        counts = None
        label = None
        if rank != 3:
            for body in tree.iter('body'):
                counts = Counter([token.lower() for token in body.text.split() if token.isalpha()])        
                if rank > 3: label = 'POS'
                else: label = 'NEG'
    return counts,label

def getCornellCounts(filename):
    counts = Counter()
    with open(filename,mode='r') as fin:
        for line in fin:
            counts += Counter([x.lower() for x in word_tokenize(line) if x.isalpha()])
    return counterString(counts)

def doIMDB(outprefix):
    negdocs = glob(os.path.join(imdb_base,'neg/*txt'))
    posdocs = glob(os.path.join(imdb_base,'pos/*txt'))
    # todo: abstract this pattern
    with codecs.open(outprefix+'.bow','w',encoding='utf-8') as fout_bow:
        with codecs.open(outprefix+'.key','w') as fout_key:
            for filename in posdocs:
                print(filename, 'POS', file=fout_key)
                print(getIMDBWordCounts(filename), file=fout_bow)
            for filename in negdocs:
                print(filename,'NEG',file=fout_key)
                print(getIMDBWordCounts(filename),file=fout_bow)

def doCornell(outprefix):
    negdocs = glob(os.path.join(cornell_base,'neg/*.txt'))
    posdocs = glob(os.path.join(cornell_base,'pos/*.txt'))
    with codecs.open(outprefix+'.bow','w',encoding='utf-8') as fout_bow:
        with codecs.open(outprefix+'.key','w') as fout_key:
            for filename in posdocs:
                print(filename,'POS', file=fout_key)
                print(getCornellCounts(filename), file=fout_bow)
            for filename in negdocs:
                print(filename,'NEG', file=fout_key)
                print(getCornellCounts(filename), file=fout_bow)
                
def doCorpusCine(outprefix):
    fails = []
    with codecs.open(outprefix+'.bow','w',encoding='utf-8') as fout_bow:
        with codecs.open(outprefix+'.key','w') as fout_key:
            for filename in glob(os.path.join(cine_base,'*.xml')):
                try:
                    counts,label = getCorpusCineCountsAndLabels(filename)
                    if counts is not None:
                        print(counterString(counts), file=fout_bow)
                        print(filename,label, file=fout_key)
                except:
                    fails.append(filename)
    return fails

def doAmazonLabelGroup(files,label,fout_key,fout_bow):
    valid = lambda string : '_' not in string and '<' not in string and '#' not in string
    for filename in files:
        with open(filename,'r') as fin:
            for line in fin:
                kvpairs = [x.split(':') for x in line.split() if valid(x)]
                counts = {i:int(j) for i,j in kvpairs if i.isalpha()}
                print(counterString(counts), file=fout_bow)
                print(label, file=fout_key)
    

def doAmazon(outprefix):
    with codecs.open(outprefix+"-unlabeled.bow",'w',encoding='utf-8') as fout_bow:
        doAmazonLabelGroup(glob(os.path.join(amazon_base,'*/unlabeled.review')),
                           'UNK',
                           None,
                           fout_bow)
    with codecs.open(outprefix+'.bow','w',encoding='utf-8') as fout_bow:
        with codecs.open(outprefix+'.key','w') as fout_key:
            doAmazonLabelGroup(glob(os.path.join(amazon_base,'*/positive.review')),
                               'POS',
                               fout_key,
                               fout_bow)
            doAmazonLabelGroup(glob(os.path.join(amazon_base,'*/negative.review')),
                               'NEG',
                               fout_key,
                               fout_bow)

