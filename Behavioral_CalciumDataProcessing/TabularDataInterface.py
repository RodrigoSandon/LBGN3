import pandas as pd
from typing import Any


class TabularDataInterface:
    def csv_to_df(self, path: str) -> pd.core.frame.DataFrame:
        pass

    def tabular_data_type() -> Any:
        pass
