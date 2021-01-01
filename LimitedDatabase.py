"""
Make a limited Dataset from another database with lots of classes
"""
import os, sys, time
import numpy as np


class LimitedDatabase:
    def __init__(self, Original_Path, Destination_path, Number_of_Classes):
        self.src = Original_Path
        self.dst = Destination_path
        self.N = Number_of_Classes
        self.path_train = self.src + '/Training'
        self.path_validation = self.src + '/Test'
        print(self.dst)

    def mkdir(self, *path):
        for p in path:
            if not os.path.exists(p):
                os.mkdir(p)
                print(f"path {p} is created")

    def link(self, src, dst):
        if not os.path.exists(dst):
            os.symlink(src, dst, target_is_directory=True)

    def createClasses(self):
        classes = []
        for cl in range(self.N):
            classes.append(np.random.choice(os.listdir(self.path_train)))
        print(classes)
        return classes

    def createDatabase(self):
        self.mkdir(self.dst)

        src_train = os.path.abspath(self.path_train)
        dst_train = os.path.abspath(self.dst + '/Training')

        src_valid = os.path.abspath(self.path_validation)
        dst_valid = os.path.abspath(self.dst + '/Validation')

        self.mkdir(dst_train, dst_valid)
        classes = self.createClasses()
        for cl in classes:
            self.link(src_train + '/' + cl, dst_train + '/' + cl)
            self.link(src_valid + '/' + cl, dst_valid + '/' + cl)

if __name__ == '__main__':
    original_path = str(sys.argv[1])
    dst_path = str(sys.argv[2])
    N = int(sys.argv[3])
    # original_path = './fruits-360/'
    # dst_path = './DATA/smallDatabase'
    print(original_path)
    # time.sleep(10)
    lm = LimitedDatabase(original_path, dst_path, N)
    lm.createDatabase()