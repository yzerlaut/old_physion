import xmltodict as xml
import os

fn = os.path.join('Bruker_xml', 'TSeries-190620-250-00-002.xml')
with open(fn, 'r') as f:
    data = xml.parse(f.read())

print(data['PVScan']['Sequence']['Frame'][0]['@absoluteTime'])
    
