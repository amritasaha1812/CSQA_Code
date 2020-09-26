Step1: Download https://drive.google.com/file/d/1ccZSys8u4F_mqNJ97OOlSLe3fjpFLhdv/view?usp=sharing and extract it (and rename the folder to lucene_dir)

Step2: Download all files from the dir. https://osu.box.com/s/8j9hy0cbi609c8kxv8cq5eeddkt7600g and place them in a dir. named transe_dir

Step3: Download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing and put it in a folder glove_dir

Step4: Download the wikidata JSONs from the link https://zenodo.org/record/4052374#.X2_KuXRKhQI and put them in a folder wikidata_dir

Step5: Put the correct (complete) paths to wikidata_dir, lucene_dir, transe_dir, glove_dir in params.py and params_test.py

Step6: In both params.py and params_test.py, use param['type_of_loss']="decoder"

Step7: Create a folder say 'Target_Model_decoder' where you want the decoder model to be dumped, and make two folders inside it ('dump' and 'model') (e.g. 'mkdir Target_Model_decoder/dump' and 'mkdir Target_Model_decoder/model')

Step8: put the params.py and params_test.py from Step 6 inside Target_Model_decoder folder

Step9: Create another version of  params.py and params_test.py, this time using param['type_of_loss']="kvmem"

Step 10: Create a folder say 'Target_Model_kvmem' where you want the kvmem model to be dumped, and make two folders inside it ('dump' and 'model') (e.g. 'mkdir Target_Model_kvmem/dump' and 'mkdir Target_Model_kvmem/model')

Step11: Download train_preprocessed.zip from https://drive.google.com/file/d/1HmLOGTV_v18grW_hXpu_s6MdogEJDM_a/view?usp=sharing and extract and put the contents (preprocessed pickle files of the train data) into Target_Model_decoder/dump and Target_Model_kvmem/dump

Step12: Download valid_preprocessed.zip from https://drive.google.com/file/d/1uoBUjjidyDks0pEUehxX-ofB5B_trdpP/view?usp=sharing and extract and put the contents (preprocessed pickle files of the valid data) into Target_Model_decoder/dump and Target_Model_kvmem/dump

Step13: Download test_preprocessed.zip from https://drive.google.com/file/d/1PMOE_jQJM_avY3MItAdEI0s3GJU_Km31/view?usp=sharing and extract and put the contents  (preprocessed pickle files of the test data) into Target_Model_decoder/dump and Target_Model_kvmem/dump

Step14: Run ./run.sh for training (the way it has been shown in the run.sh file) where the dump_dir is 'Target_Model_decoder' which you have created earlier and the data_dir is the directory containing the downloaded data

Step15: Run ./run_test.sh for testing (the way it has been shown in the run_test.sh file).

Step16: For evaluating the model separately on each question type, run the following:
./run_test.sh Target_Model_decoder verify
./run_test.sh Target_Model_decoder quantitative_count
./run_test.sh Target_Model_decoder comparative_count
./run_test.sh Target_Model_kvmem simple
./run_test.sh Target_Model_kvmem logical
./run_test.sh Target_Model_kvmem quantitative
./run_test.sh Target_Model_kvmem comparative
