from distutils.core import setup

setup(name='SlothD3MWrapper',
    version='2.0.2',
    description='A thin wrapper for interacting with New Knowledge time series tool library Sloth',
    packages=['SlothD3MWrapper'],
    install_requires=["Sloth==2.0.3"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@jg/editDeps#egg=Sloth-2.0.3"
    ],
    entry_points = {
        'd3m.primitives': [
            'time_series_segmentation.cluster.Sloth = SlothD3MWrapper:Storc'
        ],
    },
)
