import numpy as np

def aggregate_video_cluster_metrics(responses, metrics, info=None):
    """
    Aggregates video cluster metrics based on user responses.
    """
    print("[utils.py] aggregate_video_cluster_metrics() called")
    
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
        print(f"[utils.py] Clicked cluster {cluster_id}, quality: {response.get('quality', 0.0)}")

    if not is_clicked:
        metrics['cluster_watch_count_no_click'] += 1
        print("[utils.py] No clicks in this response set")

    return metrics


def write_video_cluster_metrics(metrics, add_summary_fn):
    """
    Writes average video cluster metrics using the provided summary function.
    """
    print("[utils.py] write_video_cluster_metrics() called")

    impressions = metrics.get('impression', 1)
    clicks = metrics.get('click', 0)
    quality = metrics.get('quality', 0.0)

    ctr = clicks / impressions
    add_summary_fn('CTR', ctr)
    print(f"[utils.py] CTR: {ctr}")

    if clicks > 0:
        avg_quality = quality / clicks
        add_summary_fn('AverageQuality', avg_quality)
        print(f"[utils.py] AverageQuality: {avg_quality}")

    for key, value in metrics.items():
        if key.startswith('cluster_watch_count_cluster_'):
            cluster_id = key[len('cluster_watch_count_cluster_'):]
            frac = value / impressions
            add_summary_fn(f'cluster_watch_count_frac/cluster_{cluster_id}', frac)
            print(f"[utils.py] Cluster {cluster_id} watch count frac: {frac}")

    no_clicks = metrics.get('cluster_watch_count_no_click', 0)
    no_click_frac = no_clicks / impressions
    add_summary_fn('cluster_watch_count_frac/no_click', no_click_frac)
    print(f"[utils.py] No-click watch count frac: {no_click_frac}")
