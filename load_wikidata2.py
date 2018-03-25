import json, codecs, random, pickle, traceback, logging, os, math

def load_wikidata():
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

    # wikidata_fanout_dict_list = pickle.load(open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_fanout_dict_list.pickle', 'rb'))
    # print 'Successfully loaded wikidata_fanout_dict_list pickle'

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/child_par_dict_save.json','r','utf-8') as data_file:
        child_par_dict = json.load(data_file)
    print 'Successfully loaded child_par_dict json'

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/child_all_parents_till_5_levels.json','r','utf-8') as data_file:
	child_all_parents_dict = json.load(data_file)
    print 'Successfully loaded child_all_parents_dict json'

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/filtered_property_wikidata4.json','r','utf-8') as data_file:
        prop_data = json.load(data_file)

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/par_child_dict.json','r','utf-8') as f1:
	    par_child_dict = json.load(f1)

    wikidata_remove_list = [q for q in wikidata if q not in item_data]

    wikidata_remove_list.extend([q for q in wikidata if 'P31' not in wikidata[q] and 'P279' not in wikidata[q]])

    wikidata_remove_list.extend([u'Q7375063', u'Q24284139', u'Q1892495', u'Q22980687', u'Q25093915', u'Q22980685', u'Q22980688', u'Q25588222', u'Q1668023', u'Q20794889', u'Q22980686',u'Q297106',u'Q1293664'])

    # wikidata_remove_list.extend([q for q in wikidata if q not in child_par_dict])

    for q in wikidata_remove_list:
        wikidata.pop(q,None)

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/child_par_dict_immed.json','r','utf-8') as data_file:
    	child_par_dict_immed = json.load(data_file)
    #************************ FIX for wierd parent types (wikimedia, metaclass etc.)********************************
    stop_par_list = ['Q21025364', 'Q19361238', 'Q21027609', 'Q20088085', 'Q15184295', 'Q11266439', 'Q17362920', 'Q19798645', 'Q26884324', 'Q14204246', 'Q13406463', 'Q14827288', 'Q4167410', 'Q21484471', 'Q17442446', 'Q4167836', 'Q19478619', 'Q24017414', 'Q19361238', 'Q24027526', 'Q15831596', 'Q24027474', 'Q23958852', 'Q24017465', 'Q24027515', 'Q1924819']
    stop_par_immed_list = ['Q10876391', 'Q1351452', 'Q1423994', 'Q1443451', 'Q14943910', 'Q151', 'Q15156455', 'Q15214930', 'Q15407973', 'Q15647814', 'Q15671253', 'Q162032', 'Q16222597', 'Q17146139', 'Q17633526', 'Q19798644', 'Q19826567', 'Q19842659', 'Q19887878', 'Q20010800', 'Q20113609', 'Q20116696', 'Q20671729', 'Q20769160', 'Q20769287', 'Q21281405', 'Q21286738', 'Q21450877', 'Q21469493', 'Q21705225', 'Q22001316', 'Q22001389', 'Q22001390', 'Q23840898', 'Q23894246', 'Q24025936', 'Q24046192', 'Q24571886', 'Q24731821', 'Q2492014', 'Q252944', 'Q26267864', 'Q35120', 'Q351749', 'Q367', 'Q370', 'Q3933727', 'Q4663903', 'Q4989363', 'Q52', 'Q5296', 'Q565', 'Q6540697', 'Q79786', 'Q964'] # courtsey Amrita Saha 

    ent_list = []

    for x in stop_par_list:
         ent_list.extend(par_child_dict[x])

    ent_list = list(set(ent_list))
    ent_list_resolved = [x for x in ent_list if x in child_par_dict_immed and child_par_dict_immed[x] not in stop_par_list and child_par_dict_immed[x] not in stop_par_immed_list]

    child_par_dict_val = list(set(child_par_dict.values()))
    old_2_new_pars_map = {x:x for x in child_par_dict_val}
    rem_par_list = set()

    for x in ent_list_resolved:
         child_par_dict[x] = child_par_dict_immed[x]
         old_2_new_pars_map[child_par_dict[x]] = child_par_dict_immed[x]
         rem_par_list.add(child_par_dict[x])

    ent_list_discard = list(set(ent_list) - set(ent_list_resolved))

    for q in ent_list_discard:
         par_q = None
         if q in child_par_dict:
             child_par_dict.pop(q,None)
         if q in wikidata:
             wikidata.pop(q,None)
         if q in reverse_dict:
             reverse_dict.pop(q,None)

    return wikidata, reverse_dict, prop_data, child_par_dict, child_all_parents_dict, wikidata_fanout_dict
    

