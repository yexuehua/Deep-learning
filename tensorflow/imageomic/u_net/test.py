import os
a = "./testdir/"
for path,subdir,filename in os.walk(a):
    print(path)
    print(subdir)
    print(filename)

