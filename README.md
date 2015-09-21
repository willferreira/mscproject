This repository contains the souce code for my MSc Project: "For or Against? Assessing the evidence for news headline claims.". The code is written in Python 2.7 and makes use of a number of external libraries, such as pandas, sklearn, gensim, munkres and others. To run the code from scratch, I suggest:

1. cloning the project in the normal way, i.e issuing the command: **git clone https://github.com/willferreira/mscproject.git**, at the command prompt
2. creating a new folder, called *data*,  in the top directory of the project
3. copying the contents (folders and files) from this dropbox link to the new *data* folder: https://www.dropbox.com/sh/9t7fd7xfahb0e1v/AACtdXhZmaTU9QgxZ8jL5tyVa?dl=0
4. installing the excellent anaconda distribution of Python 2.7 from continuum.io, available here: http://continuum.io/downloads 
5. creating a new Python virtual environment, by issuing the command: **conda create -n XXX anaconda python=2.7** at the command prompt (replacing XXX with whatever you want to call the environment, e.g. mscproject_py27)
6. activating the new virtual environment issuing the command: **source activate XXX** (or whatever you called it), at the command prompt
7. installing package: repoze.lru (provides a function memoize decorator) by issuing the command: **conda install repoze.lru**, at the command prompt (accept whatever package updates it proposes)
8. installing package: gensim (provides a word2vec library) by issuing the command: **conda install gensim**, at the command prompt (accept whatever package updates it proposes)
9. installing package: munkres 1.0.7 (provides an implementation of the Hungarian Algorith, used for word alignment) by:
    1. downloading the package from https://pypi.python.org/pypi/munkres/
    2. unzipping the file somewhere
    3. cd munkres-1.0.7
    4. issuing the command: **python setup.py install**, at the command prompt

You should now have all you need to run the code. The following is a description of what you can run, and what output it produces:

cd into the bin/ directory in the project. From here you can run the following:

**python run_train_test.py**

    - trains the model on the EmergentLite training data-set, and then runs the trained model on the test data-set. 
      All the features are used in the model, namely: Q,BoWHed,BoWRef,I,BoW,AlgnW2V,AlgnPPDB,RootDist,NegAlgn,SVO. The
      output consists. The output should look something like this:
      
      Feature set: ['Q', 'BoWHed', 'BoWRef', 'I', 'BoW', 'AlgnW2V', 'AlgnPPDB', 'RootDist', 'NegAlgn', 'SVO']
      >> Training classifier <<
      >> Classifying test data <<
      
      Confusion matrix:
      =================
                 for  against  observing
      for        197       11         40
      against     10       72         11
      observing   54       11        103
      
      Measures:
      =========
      accuracy: 0.7308
      
      Per class:
                  accuracy  precision     recall         F1
      for        0.7740668  0.7547893  0.7943548  0.7740668
      against    0.9155206  0.7659574  0.7741935  0.7700535
      observing  0.7721022  0.6688312  0.6130952  0.6397516







