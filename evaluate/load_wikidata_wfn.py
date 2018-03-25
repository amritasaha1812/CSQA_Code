import json, codecs, random, requests, pickle, traceback, logging, os, math

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_short_1.json','r','utf-8') as data_file:
    wikidata = json.load(data_file)
print 'Successfully loaded wikidata1'

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_short_2.json','r','utf-8') as data_file:
    wikidata2 = json.load(data_file)
print 'Successfully loaded wikidata2'

wikidata.update(wikidata2)
del wikidata2

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/items_wikidata_n.json','r','utf-8') as data_file:
    item_data = json.load(data_file)
print 'Successfully loaded items json'

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/comp_wikidata_rev.json','r','utf-8') as data_file:
    reverse_dict = json.load(data_file)
print 'Successfully loaded reverse_dict json'

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_fanout_dict.json','r','utf-8') as data_file:
    wikidata_fanout_dict = json.load(data_file)
print 'Successfully loaded wikidata_fanout_dict json'

wikidata_fanout_dict_list = pickle.load(open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_fanout_dict_list.pickle', 'rb'))
print 'Successfully loaded wikidata_fanout_dict_list pickle'

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/child_par_dict_save.json','r','utf-8') as data_file:
    child_par_dict = json.load(data_file)
print 'Successfully loaded child_par_dict json'


with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/filtered_property_wikidata4.json','r','utf-8') as data_file:
    prop_data = json.load(data_file)

wikidata_remove_list = [q for q in wikidata if q not in item_data]

wikidata_remove_list.extend([q for q in wikidata if 'P31' not in wikidata[q] and 'P279' not in wikidata[q]])

wikidata_remove_list.extend([u'Q7375063', u'Q24284139', u'Q1892495', u'Q22980687', u'Q25093915', u'Q22980685', u'Q22980688', u'Q25588222', u'Q1668023', u'Q20794889', u'Q22980686',u'Q297106',u'Q1293664'])

# wikidata_remove_list.extend([q for q in wikidata if q not in child_par_dict])

for q in wikidata_remove_list:
    wikidata.pop(q,None)
# ************************************************************************
with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_type_dict.json','r','utf-8') as f1:
    wikidata_type_dict = json.load(f1)

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_rev_type_dict.json','r','utf-8') as f1:
    wikidata_type_rev_dict = json.load(f1)

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/par_child_dict.json','r','utf-8') as f1:
    par_child_dict = json.load(f1)

with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/child_par_dict_name3.json','r','utf-8') as f1:
    child_par_dict_name_2 = json.load(f1)

# with codecs.open('dict_val/sing_sub_annot.json','r','utf-8') as f1:
# 	sing_sub_annot = json.load(f1)

# with codecs.open('dict_val/plu_sub_annot.json','r','utf-8') as f1:
# 	plu_sub_annot = json.load(f1)

# with codecs.open('dict_val/sing_obj_annot.json','r','utf-8') as f1:
# 	sing_obj_annot = json.load(f1)

# with codecs.open('dict_val/plu_obj_annot.json','r','utf-8') as f1:
# 	plu_obj_annot = json.load(f1)

# # with codecs.open('dict_val/neg_sub_annot.json','r','utf-8') as f1:
# # 	neg_sub_annot = json.load(f1)

# with codecs.open('dict_val/neg_plu_sub_annot.json','r','utf-8') as f1:
# 	neg_plu_sub_annot = json.load(f1)

# # with codecs.open('dict_val/neg_obj_annot.json','r','utf-8') as f1:
# # 	neg_obj_annot = json.load(f1)

# with codecs.open('dict_val/neg_plu_obj_annot.json','r','utf-8') as f1:
# 	neg_plu_obj_annot = json.load(f1)

# # ******************************************************************

# with codecs.open('dict_val/sing_sub_annot_wh.json','r','utf-8') as f1:
# 	sing_sub_annot_wh = json.load(f1)

# with codecs.open('dict_val/plu_sub_annot_wh.json','r','utf-8') as f1:
# 	plu_sub_annot_wh = json.load(f1)

# with codecs.open('dict_val/sing_obj_annot_wh.json','r','utf-8') as f1:
# 	sing_obj_annot_wh = json.load(f1)

# with codecs.open('dict_val/plu_obj_annot_wh.json','r','utf-8') as f1:
# 	plu_obj_annot_wh = json.load(f1)

# # with codecs.open('dict_val/neg_sub_annot_wh.json','r','utf-8') as f1:
# # 	neg_sub_annot_wh = json.load(f1)

# with codecs.open('dict_val/neg_plu_sub_annot_wh.json','r','utf-8') as f1:
# 	neg_plu_sub_annot_wh = json.load(f1)

# # with codecs.open('dict_val/neg_obj_annot_wh.json','r','utf-8') as f1:
# # 	neg_obj_annot_wh = json.load(f1)

# with codecs.open('dict_val/neg_plu_obj_annot_wh.json','r','utf-8') as f1:
# 	neg_plu_obj_annot_wh = json.load(f1)

# with codecs.open('prop_obj_90_map5.json','r','utf-8') as data_file:
#     obj_90_map = json.load(data_file)

# with codecs.open('prop_sub_90_map5.json','r','utf-8') as data_file:
#     sub_90_map = json.load(data_file)
