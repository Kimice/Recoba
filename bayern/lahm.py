import nltk
from nltk.corpus import (
    treebank,
    brown,
    sinica_treebank
)
from nltk.book import (
    text1,
    FreqDist
)

# sentence = "We don`t talk any more."
# tokens = nltk.word_tokenize(sentence)
# print tokens
# tagged = nltk.pos_tag(tokens)
# print tagged
# t = treebank.parsed_sents('wsj_0001.mrg')[0]
# t.draw()

# words_tag = brown.tagged_words(categories='news')
# print words_tag[:10]
# tagged_sents = brown.tagged_sents(categories='news')
# print tagged_sents

# print sinica_treebank.fileids()
# words = sinica_treebank.words('parsed')
# print words[:40]
# words_tag = sinica_treebank.tagged_words('parsed')
# print words_tag[:40]
# words_tag = sinica_treebank.tagged_words('parsed')
# tag_fd = nltk.FreqDist(tag for (word, tag) in words_tag)
# tag_fd.tabulate(5)

fdist1 = FreqDist(text1)
for key, value in enumerate(fdist1):
    print key, value
