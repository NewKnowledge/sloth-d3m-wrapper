import sys
import os.path
import numpy as np
import pandas

from Sloth import Sloth
from tslearn.datasets import CachedDatasets

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame

from timeseries_loader import TimeSeriesLoaderPrimitive

__author__ = 'Distil'
__version__ = '2.0.2'
__contact__ = 'mailto:jeffrey.gleason@newknowledge.io'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'GlobalAlignmentKernelKMeans', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['GlobalAlignmentKernelKMeans', 'TimeSeriesKMeans', 'DBSCAN', 'HDBSCAN'],
        description = 'type of clustering algorithm to use')
    nclusters = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=3, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'number of clusters \
        to user in kernel kmeans algorithm')
    eps = hyperparams.Uniform(lower=0, upper=sys.maxsize, default = 0.5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'maximum distance between two samples for them to be considered as in the same neigborhood, \
        used in DBSCAN algorithm')
    min_samples = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default = 5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'number of samples in a neighborhood for a point to be considered as a core point, \
        used in DBSCAN and HDBSCAN algorithms')   
    pass

class Storc(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Produce primitive's best guess for the cluster number of each series.
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "77bf4b92-2faa-3e38-bb7e-804131243a7f",
        'version': __version__,
        'name': "Sloth",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series','Clustering'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/sloth-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package': 'cython',
            'version': '0.28.5',
        },{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/sloth-d3m-wrapper.git@{git_commit}#egg=SlothD3MWrapper'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_segmentation.time_series_clustering.Sloth',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.SPECTRAL_CLUSTERING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_SEGMENTATION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : Input pandas frame where each row is a series.  Series timestamps are store in the column names.

        Returns
        -------
        Outputs
            The output is a dataframe containing a single column where each entry is the associated series' cluster number.
        """
        # setup model up
        sloth = Sloth()

        ts_loader = TimeSeriesLoaderPrimitive(hyperparams = {"time_col_index":0, "value_col_index":1, "file_col_index": None})
        inputs = ts_loader.produce(inputs = inputs).value

        # set number of clusters for k-means
        if self.hyperparams['algorithm'] == 'TimeSeriesKMeans':
            # enforce default value
            if not self.hyperparams['nclusters']:
                nclusters = 4
            else:
                nclusters = self.hyperparams['nclusters']
            labels = sloth.ClusterSeriesKMeans(inputs.values, nclusters, 'TimeSeriesKMeans')
        elif self.hyperparams['algorithm'] == 'DBSCAN':
            # enforce default value
            if not self.hyperparams['eps']:
                nclusters = 0.5
            else:
                eps = self.hyperparams['eps']
            if not self.hyperparams['min_samples']:
                min_samples = 5
            else:
                min_samples = self.hyperparams['min_samples']
                SimilarityMatrix = sloth.GenerateSimilarityMatrix(inputs.values)
                nclusters, labels, cnt = sloth.ClusterSimilarityMatrix(SimilarityMatrix, eps, min_samples)
        elif self.hyperparams['algorithm'] == 'HDBSCAN':
            # enforce default value
            if not self.hyperparams['min_samples']:
                min_samples = 5
            else:
                min_samples = self.hyperparams['min_samples']
                SimilarityMatrix = sloth.GenerateSimilarityMatrix(inputs.values)
                nclusters, labels, cnt = sloth.HClusterSimilarityMatrix(SimilarityMatrix, min_samples)
        else:
            # enforce default value
            if not self.hyperparams['nclusters']:
                nclusters = 4
            else:
                nclusters = self.hyperparams['nclusters']
            labels = sloth.ClusterSeriesKMeans(inputs.values, nclusters, 'GlobalAlignmentKernelKMeans')       

        # add metadata to output
        out_df_sloth = pandas.DataFrame(labels)
        out_df_sloth.columns = ['labels']

        # initialize the output dataframe as input dataframe (results will be appended to it)
        # out_df = d3m_DataFrame(inputs)

        sloth_df = d3m_DataFrame(out_df_sloth)
        # first column ('labels')
        col_dict = dict(sloth_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'labels'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')
        sloth_df.metadata = sloth_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        # concatentate final output frame -- not real consensus from program, so commenting out for now
        # out_df = utils_cp.append_columns(out_df, sloth_df)

        return CallResult(sloth_df)

if __name__ == '__main__':
    
    # Load data and preprocessing
    input_dataset = container.Dataset.load('file:///data/home/jgleason/D3m/datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = {"dataframe_resource":"0"})
    df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)    
    storc_client = Storc(hyperparams={'algorithm':None,'nclusters':None})#hyperparams={'algorithm':'DBSCAN','eps':0.5, 'min_samples':5})
    result = storc_client.produce(inputs = df)
    print(result.value)
    with open('sloth_predictions.csv') as file:
        file.write(str(result.value))
    
