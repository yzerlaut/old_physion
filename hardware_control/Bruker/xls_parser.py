import xml.etree.ElementTree as ET
import numpy as np

def bruker_xml_parser(file): 
    mytree = ET.parse(file)
    root = mytree.getroot()

    data = {'settings':{}}

    # for r in root[2]:
    #     for i, key in enumerate(r):
    #         print(r[i].attrib)

    settings = root[1]
    for setting in settings:
        if 'value' in setting.attrib:
            data['settings'][setting.attrib['key']] = setting.attrib['value']
        else:
            data['settings'][setting.attrib['key']] = {}
            for s in setting:
                if s.tag == 'IndexedValue':
                    if 'description' in s.attrib:
                        data['settings'][setting.attrib['key']][s.attrib['description']] = s.attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s.attrib['value']
                elif s.tag == 'SubindexedValues':
                    if len(list(s)) == 1:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s[0].attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = {}
                        for sub in s:
                            data['settings'][setting.attrib['key']][s.attrib['index']][sub.attrib['description']] = [sub.attrib['value']]
    
    frames = root[2]
    for channel in ['Ch1', 'Ch2']:
        data[channel] = {'relativeTime':[],
                         'absoluteTime':[],
                         'tifFile':[]}
    data['StartTime'] = frames.attrib['time']
    
    for x in frames:  
        if x.tag == 'Frame':
            for f in x:
                for channel in ['Ch1', 'Ch2']:
                    if f.tag == 'File' and f.attrib['channelName'] == channel:
                        data[channel]['tifFile'].append(f.attrib['filename'])
                        for key in ['relativeTime', 'absoluteTime']:
                            data[channel][key].append(float(x.attrib[key]))
                            
    for channel in ['Ch1', 'Ch2']:
        for key in ['relativeTime', 'absoluteTime']:
            data[channel][key] = np.array(data[channel][key], dtype=np.float64)
        for key in ['tifFile']:
            data[channel][key] = np.array(data[channel][key], dtype=str)
                        
    return data


if __name__=='__main__':

    import sys, os, pathlib
    
    example_file = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]),
                                'Ca_imaging', 'Bruker_xml', 'TSeries-190620-250-00-002.xml')
    data = bruker_xml_parser(example_file)
    print(data.keys())
    print(data['Ch1'].keys())
    print(data['Ch1']['absoluteTime'][-10:])
    print(data['Ch1']['tifFile'][-10:])
    import pprint
    pprint.pprint(data['settings'])
