from distutils.core import setup

setup(name='SlothD3MWrapper',
    version='2.0.2',
    description='A thin wrapper for interacting with New Knowledge time series tool library Sloth',
    packages=['SlothD3MWrapper'],
    install_requires=["Sloth>=2.0.2"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@82a1e08049531270256f38ca838e6cc7d1119223#egg=Sloth-2.0.3"
    ],
    entry_points = {
        'd3m.primitives': [
            'time_series_segmentation.cluster.Sloth = SlothD3MWrapper:Storc'
        ],
    },
)
