import nltk
import re
import copy
# Used when tokenizing words
sentence_re = r'''(?x)      # set flag to allow verbose regexps
      ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*            # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
    | \.\.\.                # ellipsis
    | [][.,;"'?():-_`]      # these are separate tokens
'''

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
#Taken from Su Nam Kim Paper...
grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
"""
chunker = nltk.RegexpParser(grammar)

d="which administrative territorial entity is the capital of united states of america and is not the location of bonekampstraat ?"
values = set([])
toks = nltk.word_tokenize(d)
postoks = nltk.pos_tag(toks)
tree = chunker.parse(postoks)
print tree
for subtree in tree.subtrees():
	if subtree==tree:
		continue
	chunk = ' '.join([x[0].strip() for x in subtree.leaves()]).strip()
	if len(chunk)<=1:
		continue
	values.add(chunk)		
print values
