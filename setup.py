import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='tradingbot',
    version='0.0',
    description='Brads crypto trading bot',
    long_description="",
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='',
    author_email='',
    url='',
    keywords='cryptos coinbase',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    extras_require={

    },
    package_data={

    },
    install_requires=[

    ],
    entry_points={
        'console_scripts': [
            'tradingbot_run = tradingbot.bin.run_bot:main',
            'tradingbot_train = tradingbot.bin.train_bot:main',
            'tradingbot_collect_data = tradingbot.bin.collect_data:main'
        ]
    },
)

