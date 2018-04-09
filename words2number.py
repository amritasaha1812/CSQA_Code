import inflect

def text2int(textnum, numwords={}):
    textnum = textnum.replace(',','')
    textnum = textnum.replace('-',' ')

    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current
'''
inf_eng = inflect.engine()
num = 1995

x = inf_eng.number_to_words(int(num))
print x
print text2int(x)

for n in range(1000000):
  if n % 100 == 0:
    print n
  n_rec = text2int(inf_eng.number_to_words(int(n)))
  try:
    assert n == n_rec
  except:
    print n
'''
