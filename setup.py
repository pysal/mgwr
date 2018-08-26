from setuptools import setup
from distutils.command.build_py import build_py

# Get __version__ from PACKAGE_NAME/__init__.py without importing the package
# __version__ has to be defined in the first line
with open('mgwr/__init__.py', 'r') as f:
    exec(f.readline())

setup(name='mgwr', #name of package
      version=version,
      description='multiscale geographically weighted regression', #short <80chr description
      url='https://github.com/pysal/mgwr', #github repo
      maintainer='Taylor M. Oshan', 
      maintainer_email='tayoshan@gmail.com', 
      test_suite = 'nose.collector',
      tests_require=['nose'],
      keywords='spatial statistics',
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5'
        'Programming Language :: Python :: 3.6'
        ],
      license='2-Clause BSD',
      packages=['mgwr'],
      install_requires=['numpy', 'scipy', 'libpysal', 'spglm', 'spreg'],
      zip_safe=False,
      cmdclass = {'build.py':build_py})
