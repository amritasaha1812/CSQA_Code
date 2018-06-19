#EXAMPLE RUN: python run_test_twostep.py <DUMP_DIR> <QUES_TYPE_ID> (QUES_TYPE_ID can be either of ‘simple’,’logical’,’quantitative’,’comparative’,’verify’,’quantitative_count’,’comparative_count’)
cp $1/params_test.py .
python run_test.py $1 $2
