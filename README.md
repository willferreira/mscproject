This repository contains the souce code for my MSc Project: "For or Against? Assessing the evidence for news headline claims.". The code is written in Python 2.7 and makes use of a number of external libraries, such as pandas, sklearn and munkres. To run the code from scratch, I suggest:

1. cloning the project in the normal way, i.e issuing the command: **git clone https://github.com/willferreira/mscproject.git**, at the command prompt
2. creating a new folder, called *data*,  in the top directory of the project
3. copying the contents (folders and files) from this dropbox link to the new *data* folder: https://www.dropbox.com/sh/9t7fd7xfahb0e1v/AACtdXhZmaTU9QgxZ8jL5tyVa?dl=0
4. installing the excellent anaconda distribution of Python 2.7 from continuum.io, available here: http://continuum.io/downloads 
5. creating a new Python virtual environment, by issuing the command: **conda create -n XXX anaconda python=2.7** at the command prompt (replacing XXX with whatever you want to call the environment, e.g. mscproject_py27)
6. activating the new virtual environment issuing the command: **source activate XXX** (or whatever you called it), at the command prompt
7. installing package: repoze.lru (provides an in-memory memoizing function decorator) by issuing the command: **conda install repoze.lru**, at the command prompt
8. installing package: munkres 1.0.7 (provides an implementation of the Hungarian Algorith, used for word alignment) by:
    1. downloading the package from https://pypi.python.org/pypi/munkres/
    2. unzipping the file somewhere
    3. cd munkres-1.0.7
    4. issuing the command: **python setup.py install**, at the command prompt

You should now have all you need to run the code. The following is a description of what you can run, and what output it produces:




