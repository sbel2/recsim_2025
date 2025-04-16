# utils.py

import numpy as np

def aggregate_video_cluster_metrics(responses, metrics, info=None):
    """
    Aggregates video cluster metrics based on user responses.

    Args:
        responses (list of dict): Each dict contains keys like 'click', 'quality', 'cluster_id'.
        metrics (dict): Dictionary to accumulate metrics.
        info (optional): Additional information (currently unused).

    Returns:
        dict: Updated metrics dictionary.
    """
    metrics['impression'] += 1
    is_clicked = False

    for response in responses:
        if not response.get('click', False):
            continue
        is_clicked = True
        metrics['click'] += 1
        metrics['quality'] += response.get('quality', 0.0)
        cluster_id = response.get('cluster_id')
        cluster_key = f'cluster_watch_count_cluster_{cluster_id}'
        metrics[cluster_key] = metrics.get(cluster_key, 0) + 1

    if not is_clicked:
        metrics['cluster_watch_count_no_click'] += 1

    return metrics

def write_video_cluster_metrics(metrics, add_summary_fn):
    """
    Writes average video cluster metrics using the provided summary function.

    Args:
        metrics (dict): Dictionary containing aggregated metrics.
        add_summary_fn (callable): Function to log or store summary metrics.
    """
    impressions = metrics.get('impression', 1)
    clicks = metrics.get('click', 0)
    quality = metrics.get('quality', 0.0)

    add_summary_fn('CTR', clicks / impressions)
    if clicks > 0:
        add_summary_fn('AverageQuality', quality / clicks)

    for key, value in metrics.items():
        if key.startswith('cluster_watch_count_cluster_'):
            cluster_id = key[len('cluster_watch_count_cluster_'):]
            add_summary_fn(f'cluster_watch_count_frac/cluster_{cluster_id}', value / impressions)

    no_clicks = metrics.get('cluster_watch_count_no_click', 0)
    add_summary_fn('cluster_watch_count_frac/no_click', no_clicks / impressions)
