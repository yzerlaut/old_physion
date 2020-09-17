import xmltodict as xml
import xml.etree.ElementTree as ET
import os

fn = os.path.join('Bruker_xml', 'TSeries-190620-250-00-002.xml')
data = xml.parse(ET.tostring(ET.parse(fn).getroot()))

print(data['PVScan']['Sequence']['Frame'][0]['@absoluteTime'])
    
