import os
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import pandas as pd
import numpy as np



def save_xml(image_name, bbox, save_dir='VOC07/Annotations', width=1609, height=500, channel=3):
 
    node_root = Element('annotation')
 
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'
 
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name
 
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
 
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
 
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
 
   
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'traffic-sign'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = '%s' % left
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = '%s' % top
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = '%s' % right
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = '%s' % bottom
 
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
 
    save_xml = os.path.join(save_dir, image_name.replace('png', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)
 
    return



def main():
    data = pd.read_table("GT-00006.csv",sep=";")
    #print(data)
    name_file=open('VOC07/ImageSets/Main/train.txt','r')
    name_file=name_file.readlines()
    i = 0
    for name in name_file:
        
        img=cv2.imread('VOC07/JPEGImages/'+name[:-1]+'.png')
        height,width  = img.shape[:2]
        name=name[:-1]+'.png'
        name2 = name[:-1] + '.ppm'
        bbox = np.array(data.loc[i, 'Roi.X1':'Roi.Y2'])
        #print(bbox)
        i+=1
        save_xml(image_name=name, bbox=bbox, save_dir='./VOC07/Annotations', width=width, height=height, channel=3)




if __name__ == "__main__":
        main()
