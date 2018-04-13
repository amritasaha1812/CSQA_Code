#EXAMPLE RUN: python run_test.py <DUMP_DIR> <QUES_TYPE_ID> (QUES_TYPE_ID can be either of 'Simple Question (Direct)','Verification (Boolean) (All)','Quantitative Reasoning (Count) (All)','Quantitative Reasoning (All)','Comparative Reasoning (Count) (All)','Comparative Reasoning (All)','Logical Reasoning (All)')
cp $1/params_test.py .
#jbsub -mem 100g -queue $3 -cores 1+1 -proj test_gen -err $1/e_test_$2_$4.txt -out $1/o_test_$2_$4.txt -require k80 /u/amritas8/anaconda/bin/python run_test.py $1 $2 $4
python run_test.py $1 $2
