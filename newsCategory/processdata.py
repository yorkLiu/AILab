# -*- coding: gbk -*-
# from lxml import etree
# import xml.etree.ElementTree as ET
import pandas as pd
import csv
#
# import sys
# reload(sys)
# sys.setdefaultencoding("gbk")
#
#
#
# xml_data = open('news_tensite_xml.xml', 'r').read().decode('gbk', 'ignore')
# xml_data = '<docs>' + xml_data + '</docs>'
#
# def xml2df(xml_data):
#     parser = etree.XMLParser(recover=True)
#     root = etree.fromstring(xml_data, parser=parser)
#     # root = ET.XML(xml_data) # element tree
#     all_records = [] #This is our record list which we will convert into a dataframe
#     for i, child in enumerate(root): #Begin looping through our root tree
#         record = {} #Place holder for our record
#         for subchild in child: #iterate through the subchildren to user-agent, Ex: ID, String, Description.
#             record[subchild.tag] = subchild.text #Extract the text create a new dictionary key, value pair
#         all_records.append(record) #Append this record to all_records.
#     return pd.DataFrame(all_records) #return records as DataFrame
#
#
# df_data = xml2df(xml_data)
# print df_data.columns
# df_data.to_csv("val.txt", columns= ['contenttitle', 'url', 'content'], encoding='utf-8', header=False, index=False, sep='\t')


stopwords = pd.read_csv('stop_words01.txt', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, header=None, names=['stopword'])
print stopwords.shape
print stopwords.head()