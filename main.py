from util.config import Config
from dataProcess import dataProcess
import time

if __name__ == '__main__':

    conf =  Config('./config/GCNCMI.conf')

    print(1)

    s=time.time()

    for i in range(0, 5):
        recSys = dataProcess(conf, i)
        recSys.execute(i)
        e = time.time()

        print("Run time: %f s" % (e - s), 'hjhjhjhjhjhjhjhj')



    pass
