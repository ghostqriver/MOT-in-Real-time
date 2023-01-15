import motmetrics as mm


def list_available_metrics():
    mot_challenge_metrics = list(mm.metrics.motchallenge_metrics)
    return mot_challenge_metrics


def populate_accumulator(gt_file_path, calculated_output_file_path):
    gt = mm.io.loadtxt(gt_file_path, fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(calculated_output_file_path, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

    return acc


def yield_metrics_from_accumulator(accumulator, metrics=None):
    if metrics is None:
        metrics = ['num_frames', 'mota', 'motp']
    mh = mm.metrics.create()

    summary = mh.compute(accumulator,
                         metrics,
                         name='acc')

    return summary


gt_file_p = open("gt.txt", "r")
gt_file = gt_file_p.read()
print(gt_file[:100])

sample_p = open("sample_output.txt", "r")
sample = sample_p.read()
print(sample[:100])

print(list_available_metrics())

accumulator = populate_accumulator("gt.txt", "sample_output.txt")

metrics_summary = yield_metrics_from_accumulator(accumulator)
print(metrics_summary)