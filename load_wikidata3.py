import json, codecs, random, pickle, traceback, logging, os, math

def load_wikidata():
    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_short.json','r','utf-8') as data_file:
        wikidata = json.load(data_file)
    print 'Successfully loaded wikidata'

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/items_wikidata_n.json','r','utf-8') as data_file:
        item_data = json.load(data_file)
    print 'Successfully loaded items json'

    # with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/comp_wikidata_rev.json','r','utf-8') as data_file:
    #     reverse_dict = json.load(data_file)
    # print 'Successfully loaded reverse_dict json'

    # with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_fanout_dict.json','r','utf-8') as data_file:
    #     wikidata_fanout_dict = json.load(data_file)
    # print 'Successfully loaded wikidata_fanout_dict json'

    # wikidata_fanout_dict_list = pickle.load(open('/dccstor/cssblr/vardaan/dialog-qa/wikidata_fanout_dict_list.pickle', 'rb'))
    # print 'Successfully loaded wikidata_fanout_dict_list pickle'

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/child_par_dict.json','r','utf-8') as data_file:
        child_par_dict = json.load(data_file)
    print 'Successfully loaded child_par_dict json'

    # wikidata = merge_dicts(wikidata1,wikidata2,wikidata3,wikidata4,wikidata5,wikidata6)
    # print 'Successfully merged the 6 dicts'

    with codecs.open('/dccstor/cssblr/vardaan/dialog-qa/filtered_property_wikidata4.json','r','utf-8') as data_file:
        prop_data = json.load(data_file)

    wikidata_remove_list = [q for q in wikidata if q not in item_data]

    wikidata_remove_list.extend([q for q in wikidata if 'P31' not in wikidata[q] and 'P279' not in wikidata[q]])

    wikidata_remove_list.extend([u'Q7375063', u'Q24284139', u'Q1892495', u'Q22980687', u'Q25093915', u'Q22980685', u'Q22980688', u'Q25588222', u'Q1668023', u'Q20794889', u'Q22980686',u'Q297106',u'Q1293664'])

    # wikidata_remove_list.extend([q for q in wikidata if q not in child_par_dict])

    for q in wikidata_remove_list:
        wikidata.pop(q,None)

    return wikidata, item_data, prop_data, child_par_dict
    

