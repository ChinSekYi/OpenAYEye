from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from .models import CLFSwitcher


class Pipe:
    def __init__(self, ct):
        self.ct = ct
        _, y = self.ct.get_Xy()
        self.classes = {
            k[0]: y.value_counts().max(numeric_only=True)
            for k, v in dict(y.value_counts()).items()
        }
        self.pipeline = Pipeline(
            [
                ("smote", SMOTE(sampling_strategy=self.classes)),
                ("clf", CLFSwitcher()),
            ]
        )

    def get_pipeline(self):
        return self.pipeline
