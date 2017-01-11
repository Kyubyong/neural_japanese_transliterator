# Neural Japanese Transliterator—can you do better than SwiftKey™ Keyboard?

In this project, we examine how well CNNs can transliterate Romaji, the romanization system for Japanese, into non-roman scripts such as Hiragana, Katakana, or Kanji, i.e., Chinese characters. The evaluation results for 1000 Japanese test sentences indicate that deep convolutional layers can quite easily and quickly learn to transliterate Romaji to the Japanese writing system though our simple model failed to outperform SwiftKey™ keyboard.

## Requirements
  * numpy >= 1.11.1
  * sugartensor >= 0.0.1.8 (pip install sugartensor)
  * regex (Enables us to use convenient regular expression posix)
  * janome (for morph analysis)
  * romkan (for converting kana to romaji)

## Background

<img src="images/swiftkey_ja.gif" width="200" align="right">
 
* The modern Japanese writing system employs three scripts: Hiragana, Katakana, and Chinese characters (kanji in Japanese).
* Hiragana and Katakana are phonetic, while Chinese characters are not.
* In the digital environment, people mostly type Roman alphabet (a.k.a. Romaji) to write Japanese. Basically, they rely on the suggestion the transliteration engine returns. Therefore, how accurately an engine can predict the word(s) the user has in mind is crucial with respect to a Japanese keyboard. 
* Look at the animation on the right. You are to type "nihongo", then the machine shows 日本語 on the suggestion bar.


## Problem Formulation
We frame the problem as a seq2seq task. (Actually this is a fun part. Compare this with my other repository: Neural Chinese Transliterator. Can you guess why I took different approaches between them?)

Inputs: nihongo。<br>
=> classifier <br>
=> Outputs: 日本語。
 
## Data
* For training, we used [Leipzig Japanese Corpus](http://corpora2.informatik.uni-leipzig.de/download.html). 
* For evaluation, 1000 Japanese sentences were collected separately. See `data/input.csv`.

## Model Architecture

We employed ByteNet style architecture (Check [Kalchbrenner et al. 2016](https://arxiv.org/pdf/1610.10099v1.pdf)). But we stacked simple convolutional layers without dilations.

## Work Flow

* STEP 1. Download [Leipzig Japanese Corpus](http://corpora2.informatik.uni-leipzig.de/downloads/jpn_news_2005-2008_1M-text.tar.gz).
* STEP 2. Extract it and copy `jpn_news_2005-2008_1M-sentences.txt` to `data/` folder.
* STEP 3. Run `build_corpus.py` to build a Romaji-Japanese parallel corpus.
* STEP 4. Run `prepro.py` to make vocabulary and training data.
* STEP 5. Run `train.py`.
* STEP 6. Run `eval.py` to get the results for the test sentences.
* STEP 7. Install the latest SwiftKey keyboard app and manually test it for the same sentences. (Luckily, you don't have to because I've done it:))

## Evaluation & Results

The evaluation metric is score. It is simply computed by subtracting levenshtein distance from the length of the true sentence. For example, the score below is 8 because the length of the ground truth is 12, and the distance between the two sentences is 4. Technically, it may not be the best choice, but I believe it suffices for this purpose.

Inputs&nbsp;&nbsp;: zuttosakinokotodakedone。<br/>
Expected: ずっと先のことだけどね。	<br/>
Got&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     : ずっと好きの時だけどね。

The training is quite fast. In my computer with a gtx 1080, the training reached the optimum in a couple of hours. Evaluations results are as follows. In both layouts, our models showed accuracy lower than SwiftKey by 0.3 points). Details are available in `results.csv`. 

| Layout | Full Score | Our Model | SwiftKey 6.4.8.57 |
|--- |--- |--- |--- |
|QWERTY| 129530 | 10890 (=0.84 acc.) | 11313 (=0.87 acc.)|


## Conclusions
* Unfortunately, our simple model failed to show better performance than the SwiftKey engine.
* However, there is still much room for improvement. Here are some ideas.
  * You can refine the model architecture or hyperparameters.
  * You can adopt a different evaluation metric.
  * As always, more data would be better.

## Note for reproducibility
* Download the pre-trained model file [here](https://drive.google.com/open?id=0B0ZXk88koS2KZGVTeUF3NVJUVWc) and  extract it to `asset/train/ckpt` folder.


	







