import sys
file = sys.argv[1]
'''
target_fw = open(sys.argv[2],'w')
for line in open(file).readlines():
	line = line.strip()
	words = line.split(' ')
	words_yes_no = []
	for w in words:
		if (w=='yes' or w=='no') and w not in words_yes_no:
			words_yes_no.append(w)
	new_line = ' and '.join(words_yes_no)
	target_fw.write(new_line.strip()+'\n')
	print line, '----->',new_line
target_fw.close()
'''
acc = 0.0
count = 0.0
goldlines = open(sys.argv[1]).readlines()
predlines = open(sys.argv[2]).readlines()
for goldline, predline in zip(goldlines, predlines):
	goldline = goldline.lower().strip()
	predline = predline.lower().strip()
	goldline = " ".join([x for x in goldline.lower().split(' ') if x in ['yes','no']])
        predline = " ".join([x for x in predline.lower().split(' ') if x in ['yes','no']])	
	if goldline==predline:
            acc=acc+1.0	
	count+=1.0
print acc/count
