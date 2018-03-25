import sys, os
from itertools import izip

ques_id = int(sys.argv[1])

model_dir = 'model_softmax_kvmem_valid_trip_unfilt'
op_dir_orig_hard = os.path.join(model_dir,'test_output_hard_%d' % ques_id)

op_dir_orig_hard_p1 = os.path.join(model_dir,'test_output_hard_%da' % ques_id)
op_dir_orig_hard_p2 = os.path.join(model_dir,'test_output_hard_%db' % ques_id)

if not os.path.exists(op_dir_orig_hard_p1):
	os.makedirs(op_dir_orig_hard_p1)

if not os.path.exists(op_dir_orig_hard_p2):
	os.makedirs(op_dir_orig_hard_p2)

which_line_ids = []
how_line_ids = []

with open(os.path.join(model_dir,'dump','test_hard_context_%d.txt' % ques_id)) as context_lines:
	for i,line in enumerate(context_lines):
		_, ques = line.split('|')
		ques_tokenized = ques.split(' ')

		if ques_tokenized[0].lower() == 'which':
			which_line_ids.append(i)
		else:
			how_line_ids.append(i)

print 'which lc = %d' % len(which_line_ids)
print 'how lc = %d' % len(how_line_ids)

for filename in os.listdir(op_dir_orig_hard):
	f1 = open(os.path.join(op_dir_orig_hard,filename),'r')
	line_list = f1.readlines()
	line_list_a = [line_list[i] for i in which_line_ids]
	line_list_b = [line_list[i] for i in how_line_ids]

	f2 = open(os.path.join(op_dir_orig_hard_p1,filename),'w')
	f3 = open(os.path.join(op_dir_orig_hard_p2,filename),'w')

	for line in line_list_a:
		f2.write(line)

	for line in line_list_b:
		f3.write(line)
	f1.close()
	f2.close()
	f3.close()

act_set_filename = os.path.join(model_dir,'dump','test_hard_active_set_%d.txt' % ques_id)
target_filename = os.path.join(model_dir,'dump','test_hard_target_%d.txt' % ques_id)

f1 = open(act_set_filename,'r')
line_list = f1.readlines()
line_list_a = [line_list[i] for i in which_line_ids]
line_list_b = [line_list[i] for i in how_line_ids]

f2 = open(os.path.join(model_dir,'dump','test_hard_active_set_%da.txt' % ques_id),'w')
f3 = open(os.path.join(model_dir,'dump','test_hard_active_set_%db.txt' % ques_id),'w')

for line in line_list_a:
	f2.write(line)

for line in line_list_b:
	f3.write(line)
f1.close()
f2.close()
f3.close()

f1 = open(target_filename,'r')
line_list = f1.readlines()
line_list_a = [line_list[i] for i in which_line_ids]
line_list_b = [line_list[i] for i in how_line_ids]

f2 = open(os.path.join(model_dir,'dump','test_hard_target_%da.txt' % ques_id),'w')
f3 = open(os.path.join(model_dir,'dump','test_hard_target_%db.txt' % ques_id),'w')

for line in line_list_a:
	f2.write(line)

for line in line_list_b:
	f3.write(line)
f1.close()
f2.close()
f3.close()