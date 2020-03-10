from setuptools import setup

setup(name='Devito-Examples',
      version='1.0',
      description="Seismic modeling and inversion examples using Devito.",
      long_description="""This package of examples provides a detailed description
                          and implementation of seismic modeling and inversion using Devito.""",
      url='https://slim.gatech.edu/',
      author="IGeorgai Institue of Technology",
      author_email='mlouboutin3@gatech.edu',
      license='MIT',
      packages=['seismic'],
      install_requires=['devito@ git+https://github.com/devitocodes/devito@master', 'ipython', 'matplotlib'])
