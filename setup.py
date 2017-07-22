from distutils.core import setup

exec(open('version.py').read())

setup(
    name='pyRCDs',
    packages=['pyrcds', 'pyrcds.tests'],
    version=__version__,
    author='Sanghack Lee',
    author_email='sanghack.lee@gmail.com'
)
