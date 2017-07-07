# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 20:38:55 2017

@author: User
"""
import os

version_of_NX_old = "\"NX\""
version_of_NX_new = "\"NX_dev\""

for f in os.listdir(os.getcwd()):
    if f.find(".pl") != -1:
        with open(f,"r") as fin, open("temp.new","w") as fout:
            for _ in fin.readlines():
                if _.find(version_of_NX_old) != -1:
                    fout.write(_.replace(version_of_NX_old,#
                                         version_of_NX_new))
                else:
                    fout.write(_)
            
        os.remove(f)
        os.rename("temp.new",f)
