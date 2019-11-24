from setuptools import setup, find_packages
from distutils.core import setup
setup(
  name = 'coat',         
  packages = find_packages(),  
  version = '0.2',      
  license='MIT',   
  download_url = 'https://github.com/moyogatomi/coat/blob/master/dist/coat-0.2.tar.gz',
  description = 'Wrapper around ndarray and opencv for rapid prototyping',   
  author = 'moyogatomi',                 
  author_email = 'moyogatomi@gmail.com',      
  url = 'https://github.com/moyogatomi/coat',   
  keywords = ['ndarray', 'opencv', 'prototyping','image processing'], 
  install_requires=["certifi==2019.9.11","chardet==3.0.4","colorama==0.4.1","idna==2.8","numpy==1.17.2","opencv-python==4.1.1.26","pafy==0.5.4","Pillow==6.2.0","pkg-resources==0.0.0","requests==2.22.0","sty==1.0.0b12","tqdm==4.36.1","urllib3==1.25.6","youtube-dl==2019.9.28","validators"],
  classifiers=[
    'Programming Language :: Python :: 3.6',     
    'Programming Language :: Python :: 3.7',
  ],
)