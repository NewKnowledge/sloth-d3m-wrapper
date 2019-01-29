from distutils.core import setup

setup(name='SlothD3MWrapper',
    version='2.0.1',
    description='A thin wrapper for interacting with New Knowledge time series tool library Sloth',
    packages=['SlothD3MWrapper'],
    install_requires=["Sloth==2.0.2"],
    dependency_links=[
        "git+https://github.com/NewKnowledge/sloth@jg/editDeps#egg=Sloth-2.0.2"
    ],
    entry_points = {
        'd3m.primitives': [
            'time_series_segmentation.time_series_clustering.Sloth = SlothD3MWrapper:Storc'
        ],
    },
)
