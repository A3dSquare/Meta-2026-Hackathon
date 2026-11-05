"""Human-readable annotations that keep demonstration reports valid JSON."""


def annotate_controller_thresholds(report, evaluation_min=32):
    """Add adjacent comments without changing raw metrics or gate results."""
    for section in ('controller_before', 'controller_after'):
        evaluation = report.get(section, {})
        trusted = set(evaluation.get('trusted_modes', []))
        for mode, result in evaluation.get('modes', {}).items():
            old = result.get('incumbent', {})
            approved = mode in trusted
            for role in ('incumbent', 'candidate'):
                metrics = result.get(role)
                if not metrics:
                    continue
                comments = {
                    'n': f'At least {evaluation_min} held-out rows are required to evaluate this mode.',
                    'labels': f'At least {max(8, evaluation_min // 4)} witnessed labels for initial approval.',
                    'accuracy': (
                        ('Current: unavailable. ' if metrics['accuracy'] is None else f'Current: {metrics["accuracy"]:.1%}. ')
                        + (f'Candidate must be at least {old["accuracy"] - .01:.1%} '
                           '(incumbent minus 1 percentage point) when sufficiently supervised.' if approved else
                           'Initial approval requires at least 75%.')
                    ),
                    'action_recall': (
                        'Order: [release, hold]. Current: '
                        + ', '.join('unavailable' if x is None else f'{x:.1%}' for x in metrics['action_recall'])
                        + '. Each must be at least 85%. Uses witnessed labels, not planner guesses (evaluation policy v3).'
                    ),
                    'action_label_counts': 'Order: [release, hold]. At least 4 labeled examples of EACH action.',
                    'human_action_label_counts': 'Human-only subset of witnessed labels; counts are frames, not independent maneuvers.',
                    'human_action_recall': 'Human-only [release, hold] recall, reported separately for diagnosis; the gate uses all witnessed action_recall.',
                    'planner_agreement': 'Diagnostic agreement with planner recommendations; does not count as witnessed action accuracy or satisfy approval.',
                    'position_rmse': 'Motion validation requires <= 0.300 blocks; lower is better.',
                    'velocity_rmse': 'Motion validation requires <= 4.000 blocks/second; lower is better.',
                    'motion_samples': f'At least {evaluation_min} motion samples AND witnessed motion under both hold and release.',
                    'unsafe_rate': (
                        f'Current: {metrics["unsafe_rate"]:.1%}. '
                        + (f'Candidate must be <= {old["unsafe_rate"] + .005:.1%} '
                           '(incumbent plus 0.5 percentage points).' if approved else
                           'Initial approval requires <= 5%; lower is better.')
                    ),
                }
                if approved:
                    comments['position_rmse'] += f' Also <= {old["position_rmse"] * 1.05 + .01:.6f} for incumbent protection.'
                    comments['velocity_rmse'] += f' Also <= {old["velocity_rmse"] * 1.05 + .1:.6f} for incumbent protection.'
                    comments['td_error'] = f'Candidate must be <= {old["td_error"] * 1.1 + .01:.6f} for incumbent protection.'
                annotated = {}
                for key, value in metrics.items():
                    if key.endswith('_comment'):
                        continue
                    annotated[key] = value
                    if key in comments:
                        annotated[key + '_comment'] = comments[key]
                result[role] = annotated
            result['thresholds_comment'] = (
                'Thresholds apply to the candidate; incumbent metrics are a reference. '
                'Passing individual metrics is not approval: coverage, incumbent protection, '
                'overall offline improvement and live comparison also apply. '
                'The saved passed/motion_verified fields are the authoritative evaluation results. '
                'Position/velocity p95 and interval_velocity_rmse are diagnostics without separate gate thresholds.'
            )
    return report
