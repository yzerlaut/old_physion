import xml.etree.ElementTree as ET
import numpy as np

def bruker_xml_parser(filename):
    """
    function to parse the xml metadata file produced by the Prairie software

    TODO:
    - find automated ways to count channels
    """
    mytree = ET.parse(filename)
    root = mytree.getroot()

    data = {'settings':{}, 'date':root.attrib['date']}
    
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
                         'depth':[],
                         'tifFile':[]}
    data['StartTime'] = frames.attrib['time']
    
    for x in frames:
        if x.tag == 'Frame':
            for f in x:
                for channel in ['Ch1', 'Ch2']:
                    if f.tag == 'File' and (channel in f.attrib['channelName']):
                        data[channel]['tifFile'].append(f.attrib['filename'])
                        for key in ['relativeTime', 'absoluteTime']:
                            data[channel][key].append(float(x.attrib[key]))
                    # depth
                    if f.tag == 'PVStateShard':
                        for d in f:
                            if d.attrib['key']=='positionCurrent':
                                for e in d:
                                    if e.attrib['index']=='ZAxis':
                                        for g in e:
                                            data[channel]['depth'].append(float(g.attrib['value']))

    # translation to numpy arrays
    for channel in ['Ch1', 'Ch2']:
        for key in ['relativeTime', 'absoluteTime']:
            print(np.unique(data[channel][key]))
            data[channel][key] = np.array(data[channel][key], dtype=np.float64)
        for key in ['tifFile']:
            data[channel][key] = np.array(data[channel][key], dtype=str)
                        
    return data


if __name__=='__main__':

    import sys, os, pathlib

    # we test it on the example file that we have in the repo:
    example_file = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]),
                                'Ca_imaging', 'Bruker_xml', 'TSeries-190620-250-00-002.xml')
    
    example_file = str('C:\\Users\\yann.zerlaut\\UNPROCESSED\\TSeries-10142021-1325-010\\TSeries-10142021-1325-010.xml')
    
    data = bruker_xml_parser(example_file)
    print(data.keys())
    for key in ['Ch1', 'Ch2']:
        print(data[key].keys())
        print(data[key]['absoluteTime'][-10:])
        print(data[key]['tifFile'][-10:])
    import pprint
    pprint.pprint(data['settings'])
