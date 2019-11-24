from setuptools import setup, find_packages
from distutils.core import setup

with open('README.md','r') as rm:
  readme = rm.read()

setup(
  name = 'coat',         
  packages = find_packages(),  
  version = '0.6',    
  license='MIT',  
  long_description=readme,
  long_description_content_type='text/markdown',
  download_url = 'https://github.com/moyogatomi/coat/blob/master/dist/coat-0.6.tar.gz',
  description = 'Wrapper around ndarray and opencv for rapid prototyping',   
  author = 'moyogatomi',                 
  author_email = 'moyogatomi@gmail.com',      
  url = 'https://github.com/moyogatomi/coat',   
  keywords = ['ndarray', 'opencv', 'prototyping','image processing'], 
  install_requires=[
"Pillow==6.2.0","numpy==1.17.2","opencv-python==4.1.1.26","pafy==0.5.4","requests==2.22.0","tqdm==4.36.1","urllib3==1.25.6","validators"],
    classifiers=[
    'Programming Language :: Python :: 3.6',     
    'Programming Language :: Python :: 3.7',
  ],
)