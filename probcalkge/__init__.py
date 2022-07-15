from calmodels import *
from calmetrics import *
from calutils import *
from caldatasets import *
from kgemodels import *
from experiment import *

import os
os.environ['AMPLIGRAPH_DATA_HOME'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'ampligraph_datasets'
    )
