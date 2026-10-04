"""Emit dataset-level classification metrics into standalone generated programs."""


def classification_metric_helpers(mode: str, names: list[str]) -> list[str]:
    return [f"    summary_mode = {mode!r}", f"    summary_names = {names!r}"] + """
    def update_summary(state, output, target):
        if summary_mode.startswith('multiclass'):
            labels = target.detach().reshape(-1).cpu()
            valid = labels != criterion.ignore_index
            labels = labels[valid]
            scores = output.detach().float().reshape(-1, output.size(-1))[valid.to(output.device)].cpu()
            scores = scores.exp() if summary_mode == 'multiclass_logprob' else scores.softmax(-1)
            predictions = scores.argmax(-1)
        else:
            labels = target.detach().reshape(-1).cpu()
            if not torch.all((labels == 0) | (labels == 1)):
                raise ValueError('Binary classification metrics require targets of 0 or 1')
            labels = labels.long()
            positive = output.detach().float().reshape(-1).cpu()
            if summary_mode == 'binary_logits':
                positive = positive.sigmoid()
            scores = torch.stack((1 - positive, positive), dim=-1)
            predictions = (positive >= 0.5).long()
        classes = scores.size(-1)
        counts = torch.stack((torch.bincount(labels[predictions == labels], minlength=classes),
                              torch.bincount(predictions, minlength=classes),
                              torch.bincount(labels, minlength=classes))).double()
        state['counts'] = counts if state['counts'] is None else state['counts'] + counts
        if 'auc' in summary_names:
            state['scores'].append(scores)
            state['labels'].append(labels)

    def finish_summary(state):
        true_positive, predicted, actual = state['counts']
        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / actual.clamp_min(1)
        f1 = 2 * true_positive / (predicted + actual).clamp_min(1)
        values = {'precision': precision, 'recall': recall, 'f1': f1, 'f1_score': f1}
        result = {name: (value.mean() if summary_mode.startswith('multiclass') else value[1]).item()
                  for name, value in values.items() if name in summary_names}
        if 'auc' in summary_names:
            scores, labels = torch.cat(state['scores']), torch.cat(state['labels'])
            class_indices = range(scores.size(-1)) if summary_mode.startswith('multiclass') else [1]
            areas = []
            for index in class_indices:
                positive = labels == index
                positives, negatives = int(positive.sum()), int((~positive).sum())
                if not positives or not negatives:
                    areas = []
                    break
                ordered, order = scores[:, index].sort()
                _, counts = torch.unique_consecutive(ordered, return_counts=True)
                end_ranks = counts.cumsum(0).double()
                ranks = torch.repeat_interleave(end_ranks - (counts.double() - 1) / 2, counts)
                area = (ranks[positive[order]].sum().item() - positives * (positives + 1) / 2) / (positives * negatives)
                areas.append(area)
            result['auc'] = sum(areas) / len(areas) if areas else None
        return result
""".strip("\n").splitlines()
