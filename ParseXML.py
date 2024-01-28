import xml.etree.ElementTree as ET
from collections import defaultdict

def get_root(filePath: str):
    tree = ET.parse(filePath)
    for _, el in enumerate(tree.iter()):
        el.tag = el.tag.split('}', 1)[1] # strip namespace
    root = tree.getroot()

    return root

def get_nodes(tree: ET.Element):
    nodes_dict = defaultdict(list)
    for node in tree.findall('.//networkStructure//nodes//node'):
        id = node.attrib.get('id')
        x = float(node[0][0].text)
        y = float(node[0][1].text)
        nodes_dict[id].append(x)
        nodes_dict[id].append(y)
        
    return nodes_dict

def get_links(tree: ET.Element):
    links_dict = defaultdict(list)
    for links in tree.findall('.//networkStructure//links'):
        for link in links:
            id = link.attrib.get('id')
            src = link[0].text
            trg = link[1].text
            links_dict[id].append(src)
            links_dict[id].append(trg)

    return links_dict

def get_demands_with_values(tree: ET.Element):
    demands_dict = defaultdict(lambda: defaultdict(list))
    demands_values = defaultdict(float)
    for demands in tree.findall('.//demands'):
        for demand in demands:
            demand_id = demand.get('id')
            demand_value = float(demand[2].text)
            demands_values[demand_id] = demand_value
            adm_paths = list(demand[3].iter("admissiblePath"))
            for adm_path in adm_paths:
                adm_path_id = adm_path.attrib.get('id')
                links = list(adm_path.iter('linkId'))
                for link in links:
                    link_id = link.text
                    demands_dict[demand_id][adm_path_id].append(link_id)

    return demands_dict, demands_values