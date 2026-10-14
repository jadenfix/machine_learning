"""Observe actual scaler fits made by sklearn stacking CV."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.preprocessing import StandardScaler


class StackingFoldTest(unittest.TestCase):
    def test_each_fold_fits_scaler_only_on_that_folds_training_rows(self):
        notebook = Path(__file__).resolve().parents[1] / 'Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb'
        source = ''.join(json.loads(notebook.read_text())['cells'][1]['source'])
        # Execute original setup/model definitions without running unrelated models.
        prefix = source.split('# --- Bagging ---')[0]
        namespace = {}
        exec(compile(prefix, str(notebook), 'exec'), namespace)
        from sklearn.ensemble import StackingClassifier
        from sklearn.model_selection import StratifiedKFold
        x, y = namespace['X_train'], namespace['y_train']
        observed = []
        original = StandardScaler.fit

        def fit(scaler, values, *args, **kwargs):
            observed.append((len(values), np.asarray(values).mean(axis=0)))
            return original(scaler, values, *args, **kwargs)

        with patch.object(StandardScaler, 'fit', fit):
            model = StackingClassifier(estimators=namespace['base_estimators'],
                                       final_estimator=namespace['meta_learner'], cv=5, n_jobs=1)
            model.fit(x, y)
        cv = StratifiedKFold(5)
        expected = [(len(train), np.asarray(x.iloc[train]).mean(axis=0))
                    for train, _ in cv.split(x, y)]
        self.assertEqual(len(observed), 18)  # 3 full-training refits + 3×5 OOF fits
        for size, mean in expected:
            count = sum(n == size and np.allclose(actual, mean, atol=1e-12)
                        for n, actual in observed)
            self.assertEqual(count, 3)
        self.assertEqual(sum(n == len(x) for n, _ in observed), 3)

if __name__ == '__main__':
    unittest.main()
