import numpy as np

from malaria import evalcore


def _gt(*boxes):
    return np.array([[*b, 0] for b in boxes], dtype=float).reshape(-1, 5)


def _pred(*rows):
    return np.array(rows, dtype=float).reshape(-1, 6)


def test_iou():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert np.allclose(evalcore.iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_each_ground_truth_box_is_matched_once():
    gt = _gt((0, 0, 10, 10))
    pred = _pred((0, 0, 10, 10, 0.9, 0), (1, 0, 10, 10, 0.8, 0))
    tp, matched = evalcore.match_image(pred, gt, 0.5)
    assert tp.tolist() == [True, False] and matched.tolist() == [0, -1]


def test_ap_perfect_empty_and_half():
    gts = {"a": _gt((0, 0, 10, 10), (20, 20, 30, 30)), "b": _gt((0, 0, 10, 10))}
    perfect = {"a": _pred((0, 0, 10, 10, 0.9, 0), (20, 20, 30, 30, 0.8, 0)), "b": _pred((0, 0, 10, 10, 0.7, 0))}
    assert evalcore.average_precision(evalcore.collect(perfect, gts, 0, 0.5)) == 1.0
    assert evalcore.average_precision(evalcore.collect({}, gts, 0, 0.5)) == 0.0
    # A confident false alarm ranked first caps precision at 1/2 for the only box.
    one = {"k": _gt((0, 0, 10, 10))}
    preds = {"k": _pred((50, 50, 60, 60, 0.9, 0), (0, 0, 10, 10, 0.8, 0))}
    assert abs(evalcore.average_precision(evalcore.collect(preds, one, 0, 0.5)) - 0.5) < 1e-9


def test_operating_point_and_image_level_metrics():
    gts = {"pos": _gt((0, 0, 10, 10), (20, 20, 30, 30)), "neg": _gt()}
    preds = {"pos": _pred((0, 0, 10, 10, 0.9, 0)), "neg": _pred((5, 5, 15, 15, 0.6, 0), (40, 40, 50, 50, 0.2, 0))}
    c = evalcore.collect(preds, gts, 0, 0.5)
    m = evalcore.at_threshold(c, 0.5)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["image_sensitivity"] == 1.0 and m["image_specificity"] == 0.0
    assert m["count_mae"] == 1.0 and m["count_bias"] == 0.0
    strict = evalcore.at_threshold(c, 0.7)
    assert strict["precision"] == 1.0 and strict["image_specificity"] == 1.0
    assert evalcore.best_f1_threshold(c) > 0.6


def test_other_classes_are_ignored():
    gts = {"a": np.array([[0, 0, 10, 10, 1]], dtype=float)}
    preds = {"a": _pred((0, 0, 10, 10, 0.9, 0))}
    c = evalcore.collect(preds, gts, 0, 0.5)
    assert c["n_gt"].sum() == 0 and c["tp"].sum() == 0


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    gts, preds = {}, {}
    for i in range(60):
        gts[str(i)] = _gt((0, 0, 10, 10))
        preds[str(i)] = _pred((0, 0, 10, 10, 0.9, 0)) if rng.random() < 0.7 else _pred()
    c = evalcore.collect(preds, gts, 0, 0.5)
    recall = evalcore.at_threshold(c, 0.5)["recall"]
    lo, hi = evalcore.bootstrap(c, 0.5, 300)["recall"]
    assert lo < recall < hi and hi - lo < 0.4


def test_recall_by_size():
    gts = {"a": _gt((0, 0, 8, 8), (20, 20, 60, 60))}
    preds = {"a": _pred((20, 20, 60, 60, 0.9, 0))}
    rows = evalcore.recall_by_size(preds, gts, 0, 0.5, 0.5, [16.0])
    assert [r["recall"] for r in rows] == [0.0, 1.0]
