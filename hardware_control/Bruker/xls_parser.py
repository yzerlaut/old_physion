import xml.etree.ElementTree as ET

def bruker_xml_parser(file): 
    mytree = ET.parse(file)
    root = mytree.getroot()

    data = {}
    
    a = {}
    settings = root[1]
    for setting in settings:
        if 'value' in setting.attrib:
            a[setting.attrib['key']] = setting.attrib['value']
        else:
            a[setting.attrib['key']] = {}
            for s in setting:
                if s.tag == 'IndexedValue':
                    if 'description' in s.attrib:
                        a[setting.attrib['key']][s.attrib['description']] = s.attrib['value']
                    else:
                        a[setting.attrib['key']][s.attrib['index']] = s.attrib['value']
                elif s.tag == 'SubindexedValues':
                    if len(list(s)) == 1:
                        a[setting.attrib['key']][s.attrib['index']] = s[0].attrib['value']
                    else:
                        a[setting.attrib['key']][s.attrib['index']] = {}
                        for sub in s:
                            a[setting.attrib['key']][s.attrib['index']][sub.attrib['description']] = [sub.attrib['value']]
    
    frames = root[2]
    b = {}
    b['Ch1'] = {}
    b['Ch1']['Relative Time'] = []
    b['Ch1']['Absolute Time'] = []
    b['Ch1']['File'] = []
    b['Ch2'] = {}
    b['Ch2']['Relative Time'] = []
    b['Ch2']['Absolute Time'] = []
    b['Ch2']['File'] = []
    b['StartTime'] = frames.attrib['time']
    
    for x in frames:  
        if x.tag == 'Frame':
            for f in x:
                if f.tag == 'File' and f.attrib['channelName'] == 'Ch1':
                    b['Ch1']['Relative Time'].append(x.attrib['relativeTime'])
                    b['Ch1']['Absolute Time'].append(x.attrib['absoluteTime'])
                    b['Ch1']['File'].append(f.attrib['filename'])
                elif f.tag == 'File' and f.attrib['channelName'] == 'Ch2':
                    b['Ch2']['Relative Time'].append(x.attrib['relativeTime'])
                    b['Ch2']['Absolute Time'].append(x.attrib['absoluteTime'])
                    b['Ch2']['File'].append(f.attrib['filename'])
    
    data = b
    data['settings'] = a      
    
    return data