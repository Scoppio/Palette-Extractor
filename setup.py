from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(  name = 'paletteExtractor',
        version = '0.1',
        description = 'Extract color palette from images',
        long_description='Uses kmeans to select the most important colors in an image and present them as a palette.',
        classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Education',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Utilities',
        ],
        keywords='Extract color palette from images',
        url='https://github.com/Scoppio/Palette-Extractor/',
        author='Lucas Coppio',
        author_email='lucascoppio@gmail.com',
        license='MIT',
        include_package_data=True,
        zip_safe=False)
