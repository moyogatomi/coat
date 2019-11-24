from setuptools import setup, find_packages
from distutils.core import setup
setup(
  name = 'coat',         # How you named your package folder (MyLib)
  packages = find_packages(),   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Wrapper around ndarray and opencv for rapid prototyping',   # Give a short description about your library
  author = 'moyogatomi',                   # Type in your name
  author_email = 'moyogatomi@gmail.com',      # Type in your E-Mail
  #url = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  #download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['ndarray', 'opencv', 'prototyping','image processing'],   # Keywords that define your package best
  install_requires=["certifi==2019.9.11","chardet==3.0.4","colorama==0.4.1","idna==2.8","numpy==1.17.2","opencv-python==4.1.1.26","pafy==0.5.4","Pillow==6.2.0","pkg-resources==0.0.0","requests==2.22.0","sty==1.0.0b12","tqdm==4.36.1","urllib3==1.25.6","youtube-dl==2019.9.28","validators"],
  classifiers=[
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
  ],
)