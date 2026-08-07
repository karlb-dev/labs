import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jspace_phase4.experiments.p4_lineage_analysis import (
    FIGURE_LAYOUT_RECT,
    reserve_footer_layout,
)


def test_lineage_analysis_reserves_footer_margin():
    figure, _ = plt.subplots(constrained_layout=True)
    try:
        reserve_footer_layout(figure)
        assert figure.get_layout_engine().get()["rect"] == (
            FIGURE_LAYOUT_RECT)
        assert FIGURE_LAYOUT_RECT[1] > 0
    finally:
        plt.close(figure)
