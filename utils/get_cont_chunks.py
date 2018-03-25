import nltk, re, copy

s1 = 'washington, d.c., new york city'
s2 = 'Which administrative territorial entity is the capital of United States of America and is not the location of Bonekampstraat ?'
s3 = 'Epernay, hautvillers, damery'
s4 = 'what about la vicogne?'
s5 = 'vauciennes, Epernay, damery, hautvillers, le petit-quevilly, cumieres'
s6 = 'which type of french administrative division shares border with mardeuil ?'
s7 = 'metro-goldwyn-mayer, marvel studios, bad robot productions'
s8 = 'which work of art were produced by metro-goldwyn-mayer, marvel studios and bad robot productions ?'
s9 = 'amsterdam, athens, barcelona, beijing, beirut, bethlehem, chicago, damascus, domodedovo, famagusta, istanbul, ljubljana, los angeles, madrid, mexico city, moscow, naples, nicosia, rabat, reggio calabria, rio de janeiro, sofia, washington, d.c., boston, montreal, genoa, florence, lisbon, cali, prague, warsaw, tbilisi, havana, cusco, atlanta, belgrade'
s10 = 'which administrative territorial entity is a sister city of those administrative territorial entities ?'
s11 = 'Oldham Athletic A.F.C., Kilmarnock F.C.'

s12 = 'Which administrative territorial entity is the capital of United States of America and is not the location of Bonekampstraat ?|washington, d.c., new york city'
s13 = 'which type of french administrative division shares border with mardeuil ?|vauciennes, Epernay, damery, hautvillers, le petit-quevilly, cumieres'
s14 = 'which administrative territorial entity is a sister city of those administrative territorial entities ?|amsterdam, athens, barcelona, beijing, beirut, bethlehem, chicago, damascus, domodedovo, famagusta, istanbul, ljubljana, los angeles, madrid, mexico city, moscow, naples, nicosia, rabat, reggio calabria, rio de janeiro, sofia, washington, d.c., boston, montreal, genoa, florence, lisbon, cali, prague, warsaw, tbilisi, havana, cusco, atlanta, belgrade'
s15 = 'which work of art were produced by metro-goldwyn-mayer, marvel studios and bad robot productions ?|Oldham Athletic A.F.C., Kilmarnock F.C.'
s16 = 'member of the European Parliament, Member of the Chamber of Deputies of the Parliament of the Czech Republic'
s17 = 'Which administrative territorial entity is the capital of United States of America and is not the location of Bonekampstraat ?'

text = s17

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

# values = set([])
toks = nltk.word_tokenize(text)
postoks = nltk.pos_tag(toks)
tree = chunker.parse(postoks)
# print tree
super_list = [w for w,t in tree.leaves()]
return_dict = {}

for subtree in tree.subtrees():
    # print subtree
    if subtree==tree:
      continue
    chunk_list = [x[0].strip() for x in subtree.leaves()]
    chunk = ' '.join(chunk_list).strip()
    if len(chunk)<=1:
      continue
    if chunk not in return_dict:
        return_dict[chunk] = chunk_list
    # values.add(chunk)
