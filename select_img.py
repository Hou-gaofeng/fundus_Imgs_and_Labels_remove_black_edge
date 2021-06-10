import os

imglist = os.listdir('select_label')

f = open('test.list', 'w')
for line in imglist:
    line = os.path.splitext(line)[0]
    f.write(line + '\n')
f.close()