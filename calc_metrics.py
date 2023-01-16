import motmetrics as mm
import matplotlib.pyplot as plt
import numpy as np


def list_available_metrics():
    mot_challenge_metrics = list(mm.metrics.motchallenge_metrics)
    return mot_challenge_metrics


def populate_accumulator(gt_file_path, calculated_output_file_path):
    gt = mm.io.loadtxt(gt_file_path, fmt="mot15-2D", min_confidence=1)
    ts = mm.io.loadtxt(calculated_output_file_path, fmt="mot15-2D")
    return mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)


def calculate_metrics(metrics=None, all_metrics=False):
    accumulator = populate_accumulator("gt.txt", "sample_output.txt")
    if metrics is None:
        metrics = ['num_frames', 'mota', 'motp']
    mh = mm.metrics.create()

    if all_metrics:
        summary = mh.compute(accumulator,
                             name='acc')
    else:
        summary = mh.compute(accumulator,
                             metrics=metrics,
                             name='acc')
    return summary, mh


def yield_metrics_string_from_accumulator(all_metrics=False):
    summary, mh = calculate_metrics(all_metrics)
    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    return str_summary


def plot_results(original, enhanced):
    data = [original, enhanced]
    X = np.arange(len(original))
    plt.bar(X + 0.00, data[0], color='b', width=0.25)
    plt.bar(X + 0.25, data[1], color='g', width=0.25)
    plt.xlabel('Metrics')
    plt.xticks(np.arange(len(original)) + 0.125, ['mota', 'motp'])
    plt.title('Scores')
    plt.show()


gt_file_p = open("gt.txt", "r")
gt_file = gt_file_p.read()
print(gt_file[:100])

sample_p = open("sample_output.txt", "r")
sample = sample_p.read()
print(sample[:100])

print(list_available_metrics())

metrics_summary, _ = calculate_metrics(all_metrics=True)

mota_motp_metrics = [metrics_summary['mota']['acc'], metrics_summary['motp']['acc']]
plot_results(mota_motp_metrics, mota_motp_metrics)
