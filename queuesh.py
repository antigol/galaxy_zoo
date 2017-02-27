# pylint: disable=C,R,no-member
import os
from sys import argv
from time import sleep
import shutil
import subprocess
import glob

def main(dirpath):
    if not os.path.isdir(os.path.join(dirpath, 'finished')):
        os.makedirs(os.path.join(dirpath, 'finished'))

    while True:
        sleep(1.0)
        
        if os.path.isfile(os.path.join(dirpath, 'stop')):
            return

        if os.path.isfile(os.path.join(dirpath, 'pause')):
            continue
        
        entries = sorted(glob.glob(os.path.join(dirpath, '*.sh')))
        
        if len(entries) == 0:
            continue

        current_path = entries[0]
        name = os.path.basename(current_path)

        running_path = os.path.join(dirpath, 'RUNNING_{}'.format(name))
                
        shutil.move(current_path, running_path)

        p = subprocess.Popen(['bash', running_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        rc = p.wait()

        if rc != 0:
            with open(os.path.join(dirpath, name + '.out'), 'wb') as f:
                f.write(p.stdout.read())

        shutil.move(running_path, os.path.join(dirpath, 'finished', name))


if __name__ == '__main__':
    main(argv[1])
