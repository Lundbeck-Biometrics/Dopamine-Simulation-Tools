# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:00:32 2018

@author: jakd
"""

from DopamineToolbox import AnalyzeSpikesFromFile

"The file to be opened"
chrisfile = "M:/Python/Datafiles/644301/644297.txt"
#chrisfile = "M:/Python/Datafiles/644301/632822.txt"
RES = AnalyzeSpikesFromFile(chrisfile, pre_run = 10, tmax = 10)

print('printing results:')
print(RES)