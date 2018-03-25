cp $1/params.py .
jbsub -mem 350g -queue $2 -err $1/e.txt -out $1/o.txt -require k80 python run_model.py $1 $3
