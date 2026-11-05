"""Publication formatting and data-package regression checks."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from evidence import Evidence
from reproduce_paper import ensure_reference
from reporting import tables, figures
from matplotlib.figure import Figure


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_reference()
        cls.evidence = Evidence(ROOT/'data/reference')

    def test_article_table_display(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            tables(self.evidence, folder)
            main = (folder/'table1.tex').read_text()
            self.assertIn(r'\begin{tabular}{lcccc}', main)
            self.assertIn(r'$\mathbf{119.5}\,(\pm23.4)$', main)
            self.assertIn(r'Fisher-UCB, matched ($\beta=\sqrt2$)', main)
            boundary = (folder/'table3.tex').read_text()
            self.assertIn(r'\begin{tabular}{llrrrrr}', boundary)
            self.assertIn(r'$Q^\star\approx1056$', boundary)
            self.assertIn('1600 & --- & --- & --- & ---', boundary)
            self.assertIn('363.1 & 22.6 & 1587 & 2997 & 39/200', boundary)
            self.assertIn('KL-UCB (vectorized Newton)', (folder/'table4.tex').read_text())

    def test_article_figure_styles(self):
        captured = {}
        def capture(figure, destination, **kwargs):
            captured[Path(destination).name] = figure
        with tempfile.TemporaryDirectory() as temporary, patch.object(Figure, 'savefig', capture):
            figures(self.evidence, Path(temporary))
        self.assertEqual(len(captured), 6)
        for name in ['regret_curves.pdf', 'binomial_curves_n100.pdf', 'scaling_regret_k100.pdf']:
            for ax in captured[name].axes:
                self.assertEqual(ax.title.get_fontsize(), 10)
                self.assertEqual([line.get_linestyle() for line in ax.lines], ['-.', '--', '-'])
        poisson = captured['poisson_curves.pdf']
        for ax in poisson.axes:
            self.assertTrue(all(line.get_linestyle()=='-' for line in ax.lines))
            self.assertTrue(all(band.get_alpha()==.12 for band in ax.collections))
            self.assertAlmostEqual(ax.get_legend().get_texts()[0].get_fontsize(), 8*12/11)
        horizon = captured['horizon_scaling.pdf']
        self.assertEqual(horizon.axes[1].lines[-1].get_color(), 'black')
        for ax in [*horizon.axes, *captured['scaling_runtime_combined.pdf'].axes]:
            self.assertTrue(all(line.get_linestyle()=='--' for line in ax.get_xgridlines()))
            self.assertTrue(all(line.get_alpha()==.3 for line in ax.get_xgridlines()))

    def test_reference_package_contains_only_public_data(self):
        private = re.compile(r'/home/|/Users/|[A-Z]:\\Users\\|\bv\d+\s+sequential\b', re.I)
        with zipfile.ZipFile(ROOT/'data/reference.zip') as archive:
            for member in archive.namelist():
                self.assertTrue(member.endswith(('.json', '.csv', '.npz')), member)
                if member.endswith('.json'):
                    text = archive.read(member).decode()
                    json.loads(text)
                    self.assertIsNone(private.search(text), member)


if __name__ == '__main__':
    unittest.main()
