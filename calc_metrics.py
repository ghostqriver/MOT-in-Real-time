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


def get_summary_and_mh(gt_file, output_file, metrics, all_metrics=False):
    accumulator = populate_accumulator(gt_file, output_file)
    mh = mm.metrics.create()

    if all_metrics:
        summary = mh.compute(accumulator,
                             name='acc')
    else:
        summary = mh.compute(accumulator,
                             metrics=metrics,
                             name='acc')
    return summary, mh


def calculate_metrics_for_output(gt_file, output_file, metrics=None, all_metrics=False):
    if metrics is None:
        metrics = ['num_frames', 'mota', 'motp']

    summary, _ = get_summary_and_mh(gt_file, output_file, metrics, all_metrics)

    if all_metrics:
        metrics = list_available_metrics()

    metric_list = []
    for metric in metrics:
        metric_list.append(summary[metric]['acc'])

    return metric_list


def yield_metrics_summary_from_accumulator(gt_file, output_file, metrics):
    summary, mh = get_summary_and_mh(gt_file, output_file, metrics)
    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    return str_summary


def plot_results(original_model_results, fps_enhanced_model_results, metric_names):
    data = [original_model_results, fps_enhanced_model_results]
    X = np.arange(len(original_model_results))
    plt.title('Scores')
    plt.bar(X + 0.00, data[0], color='b', width=0.25)
    plt.bar(X + 0.25, data[1], color='g', width=0.25)
    plt.xlabel('Metrics')
    plt.xticks(np.arange(len(original_model_results)) + 0.125, metric_names)
    plt.show()


gt_file_p = open("gt.txt", "r")
gt_file = gt_file_p.read()
print(gt_file[:100])

sample_p = open("sample_output.txt", "r")
sample = sample_p.read()
print(sample[:100])

print(yield_metrics_summary_from_accumulator("gt.txt", "sample_output.txt", list_available_metrics()))

plot_results(calculate_metrics_for_output("gt.txt", "sample_output.txt", all_metrics=True),
             calculate_metrics_for_output("Yizhi_pred_outputs/gt_processed_drop_each_frame_2.txt",
                                          "Yizhi_pred_outputs/2023_01_19_12_00_49.txt", all_metrics=True),
             list_available_metrics())
