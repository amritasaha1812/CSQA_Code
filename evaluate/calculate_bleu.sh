~/anaconda/bin/python calculate_bleu.py $1 $2 $3 $4 "mem" > $2/test_output_$3_$4/twostep_pred_sent_replaced_from_mem.txt
echo $2/test_output_$3_$4/twostep_pred_sent_replaced_from_mem.txt
perl multi_bleu.pl -lc $2/test_output_$3_$4/gold_resp.txt < $2/test_output_$3_$4/twostep_pred_sent_replaced_from_mem.txt
~/anaconda/bin/python calculate_bleu.py $1 $2 $3 $4 "kb" > $2/test_output_$3_$4/twostep_pred_sent_replaced_from_kb.txt
echo $2/test_output_$3_$4/twostep_pred_sent_replaced_from_kb.txt
perl multi_bleu.pl -lc  $2/test_output_$3_$4/gold_resp.txt < $2/test_output_$3_$4/twostep_pred_sent_replaced_from_kb.txt
