import json
import os
import scipy.io

"""
Solo para pruebas de concepto
Entrenar con aircraft, CUB, dtd, fungi, vgg_flower
Test con traffic
"""

class DataLoader(object):
    def __init__(self, path, dataset, path_split, split):
        #Me faltan ImageNet y omniglot
        datasets ={
            'aircraft': ['fgvc-aircraft-2013b', self.readAircraft],
            'cu_birds': ['CUB_200_2011', self.readCUB],
            'dtd': ['dtd', self.readDTD],
            'fungi': ['fungi', self.readFungi],
            'quickdraw': ['quickdraw', self.readQuickdraw],
            'traffic_sign': ['traffic_sign', self.readTraffic],
            'vgg_flower': ['VGGFlower', self.readVGGFlower],
            #'mscoco': 'mscoco'
        }

        self.path_data = os.path.join(path, datasets[dataset][0])
        self.path_split = path_split

        self.split = split

        self.classes = self.readJson()

        self.data, self.target = datasets[dataset][1]()

    def readJson(self):
        with open(self.path_split, 'r') as f:
            datastore = json.load(f)

        return datastore[self.split]

    def readAircraft(self):
        img_names = []
        img_target = []

        path = os.path.join(self.path_data,'data','images_variant_trainval.txt')

        with open(path, 'r') as f:
            for row in f.readlines():
                img_id, variant = row.split(' ', 1)
                if variant in self.classes:
                    img_names.append(os.path.join(self.path_data,'data','images',img_id+'.jpg'))
                    img_target.append(self.classes.index(variant))

        return img_names, img_target

    def readCUB(self):
        img_names = []
        img_target = []

        for cl in os.listdir(os.path.join(self.path_data, 'train')):
            if cl in self.classes:
                for img in os.listdir(os.path.join(self.path_data, 'train', cl)):
                    img_names.append(os.path.join(self.path_data, 'train', cl, img))
                    img_target.append(self.classes.index(cl))

        return img_names, img_target

    def readDTD(self):
        img_names = []
        img_target = []

        for cl in os.listdir(os.path.join(self.path_data, 'images')):
            if cl in self.classes:
                for img in os.listdir(os.path.join(self.path_data, 'images', cl)):
                    img_names.append(os.path.join(self.path_data, 'images', cl, img))
                    img_target.append(self.classes.index(cl))

        return img_names, img_target

    def readFungi(self):
        img_names = []
        img_target = []

        for cl in os.listdir(os.path.join(self.path_data, 'images')):
            c = cl.replace("'", "").replace('_','.',1).replace('_',' ')
            if c in self.classes:
                for img in os.listdir(os.path.join(self.path_data, 'images', cl)):
                    img_names.append(os.path.join(self.path_data, 'images', cl, img))
                    img_target.append(self.classes.index(c))

        return img_names, img_target

    def readQuickdraw(self):
        img_names = []
        img_target = []

        for cl in os.listdir(os.path.join(self.path_data)):
            c = cl.replace("'", "")
            if c in self.classes:
                img_names.append(os.path.join(self.path_data, cl)) #.npy
                img_target.append(self.classes.index(c))

        return img_names, img_target

    def readTraffic(self):
        img_names = []
        img_target = []

        for cl in os.listdir(os.path.join(self.path_data, 'GTSRB','Final_Training','Images')):
            if int(cl) in self.classes:
                for img in os.listdir(os.path.join(self.path_data, 'GTSRB','Final_Training','Images', cl)):
                    img_names.append(os.path.join(self.path_data, 'GTSRB','Final_Training','Images', cl, img))
                    img_target.append(self.classes.index(int(cl)))

        return img_names, img_target

    def readVGGFlower(self):
        img_names = []
        img_target = []

        labels = scipy.io.loadmat(os.path.join(self.path_data, 'imagelabels.mat'))['labels'][0]

        for i,elem in enumerate(labels):
            if elem in self.classes:
                name_img = 'image_{0:05d}.jpg'.format(i+1)
                
                img_names.append(os.path.join(self.path_data, 'jpg', name_img))
                img_target.append(self.classes.index(elem))

        return img_names, img_target

    def __len__(self):
        return len(self.data)

def main():
    path = '/mnt/nas/GrimaRepo/datasets/'

    datasets = ['aircraft', 'cu_birds', 'dtd', 'fungi', 'quickdraw', 'traffic_sign', 'vgg_flower']

    for elem in datasets:
        for split in ['train', 'valid', 'test']:
            temp = DataLoader(path, elem, os.path.join('data', elem+'_splits.json'), split)
            print(len(temp))

if __name__ == '__main__':
    main()