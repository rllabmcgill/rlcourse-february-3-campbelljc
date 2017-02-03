import re, os, simplejson

DIR_CHAR = '\\' if os.name == 'nt' else '/'

def append(arg, file, mode="a+"):
    with open(file+".txt", mode) as rfile:
        rfile.write(str(arg) + "\n")

def write_list(arg, file, mode='w'):
    f = open(file+'.txt', mode)
    simplejson.dump(list(arg), f)
    f.write('\n')
    f.close()
    
def read_list(file):
    f = open(file+'.txt', 'r')
    lst = simplejson.load(f)
    f.close()
    return lst

def read_line_list(file, ignore=[], load_float=True):
    file = file + ".txt" if not ".txt" in file else file
    if not os.path.isfile(file):
        return []
    f = open(file, 'r')
    lst = []
    for line in f:
        if f not in ignore:
            lst.append(float(line) if load_float else line)
    f.close()
    return lst
