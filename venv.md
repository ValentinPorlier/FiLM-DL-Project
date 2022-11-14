# Virtual env instructions

Install `virtualenvwrapper` with `pip install virtualenvwrapper`. Then run in a Terminal
```
# We want to regularly go to our virtual environment directory
$ echo 'export WORKON_HOME=~/.virtualenvs' >> .bash_profile
# If in a given virtual environment, make a virtual environment directory
# If one does not already exist
$ echo 'mkdir -p $WORKON_HOME' >> .bash_profile
# Activate the new virtual environment by calling this script
$ echo '. PYTHON/bin/virtualenvwrapper.sh' >> .bash_profile
```
or
```
# We want to regularly go to our virtual environment directory
$ echo 'export WORKON_HOME=~/.virtualenvs' >> .zprofile
# If in a given virtual environment, make a virtual environment directory
# If one does not already exist
$ echo 'mkdir -p $WORKON_HOME' >> .zprofile
# Activate the new virtual environment by calling this script
$ echo '. PYTHON/bin/virtualenvwrapper.sh' >> .zprofile
```
where `PYTHON` is the location of your python installation. Then refresh your Terminal or open a new Terminal window.

Create you environments with `mkvirtualenv`, switch between them or list them with `workon`, deactivate them with `deactivate`.

Inside the directory of your project you can run 
```
mkvirtualenv $(basename $(pwd))
```
to create a virtual env with the same name as your directory, so when you are inside this directory you can simply activate the corresponding environment with 
```
workon .
```

You can then run `pip install -r requirements.txt` to install your project dependencies and/or `ipython kernel install --user --name=projectname` to add your virtual env to your jupyter kernels.