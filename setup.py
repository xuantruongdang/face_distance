from cx_Freeze import setup, Executable

setup(name = "cal_dis_face" ,
      version = "0.1" ,
      description = "" ,
      executables = [Executable("source.py")])