#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Before running this code, make sure that you've downloaded Leipzig Japanese Corpus 
(http://corpora2.informatik.uni-leipzig.de/downloads/jpn_news_2005-2008_1M-text.tar.gz)
Extract and copy the `jpn_news_2005-2008_1M-sentences.txt` to `data/` folder.

This code should generate a file which looks like this:
1    enouedehashuuridets...。    絵の上では修理で使っ...。

In each line, the id, romaji, and a japanese sentence are separated by a tab.
Created in Jan. 2017, kyubyong. kbpark.linguist@gmail.com
www.github.com/kyubyong/neural_japanese_transliterator
"""
from __future__ import print_function 
import codecs
import os
import regex # pip install regex
import romkan # pip install romkan
from janome.tokenizer import Tokenizer # pip install janome

def clean(text):
    # For simplicity, character other than the allowed ones are removed.
    text = regex.sub(u"[^\p{Han}\p{Hiragana}\p{Katakana}ー]", "", text)
    return text

def get_romaji(sent):
    t = Tokenizer()
    readings = ""
    for token in t.tokenize(sent):
        surface = regex.split("[\t,]", str(token).decode('utf8'))[0]
        reading = regex.split("[\t,]", str(token).decode('utf8'))[-2]
        reading = surface if reading == "*" else reading
        readings += reading
    romaji = romkan.to_roma(readings)
    return romaji

def annotate():
    if not os.path.exists("preprocessed"): os.mkdir("preprocessed")
    with codecs.open("preprocessed/ja.tsv", 'w', 'utf-8') as fout:
        with codecs.open("data/jpn_news_2005-2008_1M-sentences.txt", 'r', 'utf-8') as fin:
            i = 1
            while 1:
                line = fin.readline()
                if not line: break
                
                try:
                    idx, sent = line.strip().split("\t")
                    sent = clean(sent)
                    romaji = get_romaji(sent)
                    fout.write(u"{}\t{}\t{}\n".format(idx, romaji, sent))
                except:
                    continue # it's okay as we have a pretty big corpus!
                
                if i % 1000 == 0: print(i,)
                i += 1

if __name__ == "__main__":
    annotate()
    print("Done")
