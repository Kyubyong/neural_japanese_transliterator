# coding: utf-8
#!/usr/bin/python2
import codecs
import lxml.etree as ET
import os
import regex
import romkan 
from janome.tokenizer import Tokenizer; t = Tokenizer()

fname = "E:/ja/jawiki-20161201-pages-articles-multistream.xml"    

def clean_text(text):
    # Common
    text = regex.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = regex.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = regex.sub("&[a-z]+;", "", text) # remove html entities
    text = regex.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = regex.sub("(?s){.+?}", "", text) # remove markup tags
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = regex.sub("[']{5}", "", text) # remove italic+bold symbols
    text = regex.sub("[']{3}", "", text) # remove bold symbols
    text = regex.sub("[']{2}", "", text) # remove italic symbols
    
    text = regex.sub(u"[^\r\n\p{Han}\p{Hiragana}\p{Katakana}\dー。！？]", "", text)
    
    # Common
    text = regex.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    return text

def sentence_segment(text):
    '''
    Args:
      text: A string. A unsegmented paragraph.
    
    Returns:
      A list of sentences.
    '''
    sents = regex.sub(u"([。！？])", r"\1 ", text)
    return sents.split()
        
def align(sent):
    '''
    Args:
      sent: A string. A sentence.
    
    Returns:
      A list of words.
    '''
    romaji, surface = '', ''
    for token in t.tokenize(u'すもももももももものうち'):
        surface += regex.split("[\t,]", str(token).decode('utf8'))[0]
        reading = regex.split("[\t,]", str(token).decode('utf8'))[-2]
        romaji += romkan.to_roma(reading)

    return romaji, surface

def build_corpus():
    with codecs.open("E:/ja/ja.txt", 'w', 'utf-8') as fout:
        i = 1
        j = 1
        ns = "{http://www.mediawiki.org/xml/export-0.10/}" # namespace
        for _, elem in ET.iterparse(fname, tag=ns+"text"):
            running_text = elem.text
            try:
                running_text = clean_text(running_text)
                sents = sentence_segment(running_text)
                for sent in sents:
                    
                    if sent is not None and 10 < len(sent) < 30:
                        romaji, surface = align(sent)
                        fout.write(u"{}\t{}\n".format(romaji, surface))
                                
            except:
                continue # it's okay as we have a pretty big corpus!
            elem.clear() # We need to save memory!
            if i % 1000 == 0: 
                print i,
                fsize = os.path.getsize("E:/zh/zh.txt")
                if fsize > 10000000:
                    break
            i += 1

if __name__ == "__main__":
    build_corpus()
    
    print "Done"