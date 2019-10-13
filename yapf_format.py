'''format a file'''
from yapf.yapflib.yapf_api import FormatFile  # reformat a file
FormatFile('common.py', in_place=True)
FormatFile('loader1.py', in_place=True)
FormatFile('utils.py', in_place=True)
<<<<<<< HEAD
=======
FormatFile('experiment2.py', in_place=True)
>>>>>>> 9e312e1027e965a313dbbf343530904e7eb4ff80
