from distutils.core import setup

setup(name='SlothD3MWrapper',
    version='2.0.1',
    description='A thin wrapper for interacting with New Knowledge time series tool library Sloth',
    packages=['SlothD3MWrapper'],
    install_requires=["Sloth==2.0.2"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@fafa4857f02c91c36c22e0d3efb16ab90d47e62d#egg=Sloth-2.0.2"
    ],
    entry_points = {
        'd3m.primitives': [
            'distil.Sloth.cluster = SlothD3MWrapper:Storc'
        ],
    },
)
