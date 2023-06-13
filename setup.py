from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(filepath:str)->List[str]:
    '''
    This function returns a list of requirements
    '''
    requirements = []
    with open(filepath, 'r') as file_obj:
        requirements = file_obj.readlines()
        [req.replace('\n','') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Howard Gan',
    author_email='ganhaozhan@live.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
