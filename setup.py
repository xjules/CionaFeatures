from setuptools import setup
import cionafeatures

_version = '0.0.1'
_requirements = []
with open('requirements.txt','r') as f:
	_requirements = [line.strip() for line in f]

setup(
	name='cionafeatures',
	version=_version,
	author='Julius Parulek (Equinor), SARS',
	email='parulek@gmail.com',
	install_requirements=_requirements,
	#test_suite='tests'
)