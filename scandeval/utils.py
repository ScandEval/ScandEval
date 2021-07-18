import warnings
from datasets.utils.logging import _reset_library_root_logger as shh_datasets
from transformers.utils.logging import set_verbosity_error as shh_transformers


warnings.filterwarnings(
    'ignore',
    module='torch.nn.parallel*',
    message=('Was asked to gather along dimension 0, but all input '
             'tensors were scalars; will instead unsqueeze and return '
             'a vector.'),
)
warnings.filterwarnings(
    'ignore',
    module='seqeval*',
    message=('UndefinedMetricWarning: Precision and F-score are '
             'ill-defined and being set to 0.0 in labels with no '
             'predicted samples. Use `zero_division` parameter to '
             'control this behavior. Was asked to gather along '
             'dimension 0, but all input tensors were scalars; '
             'will instead unsqueeze and return a vector.'),
)

shh_datasets()
shh_transformers()
