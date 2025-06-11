"""
Model-specific params
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool

preprocess_params = []

train_params = [
    {"name": "max_depth",
     "type": int,
     "default": 6,
     "help": "Tree HP. Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit and aggressively consumes memory. Range [1, inf]."
    },
    {"name": "min_child_weight",
     "type": int,
     "default": 1,
     "help": "Tree HP. Minimum sum of instance weight (hessian) needed in a child. The larger min_child_weight is, the more conservative the algorithm will be. Range [0, inf]."
    },
    {"name": "subsample",
     "type": float,
     "default": 1,
     "help": "Tree HP. Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration. Range [0, 1]."
    },
    {"name": "colsample_bytree",
     "type": float,
     "default": 1,
     "help": "Tree HP. Fraction of columns used for each tree construction. Lowering this value can prevent overfitting by training on a subset of the features. Range [0, 1]."
    },
    {"name": "gamma",
     "type": int,
     "default": 0,
     "help": "Learning HP. Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. Range [0, inf]."
    },
    {"name": "lambda",
     "type": float,
     "default": 1,
     "help": "Learning HP. L2 regularization term on weights. Increasing this value will make model more conservative. Range [0, 1]."
    },
    {"name": "alpha",
     "type": float,
     "default": 0,
     "help": "Learning HP. L1 regularization term on weights. Increasing this value will make model more conservative Range [0, 1]."
    },
]

infer_params = []