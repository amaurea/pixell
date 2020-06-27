import os,sys

vdict = {"3.6":"1.13.3","3.7":"1.14.5","3.8":"1.17.3"}
version = ".".join(map(str, sys.version_info[:2]))
np_version = vdict[version]
cmd = f"pip3 install numpy=={np_version} -vvv"
print("======================================================================")
print("Running old-numpy install command: ", cmd)
print("======================================================================")
os.system(cmd)
