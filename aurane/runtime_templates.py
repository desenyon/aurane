"""Runtime helpers embedded into generated, standalone PyTorch programs."""

ATTENTION_METHOD = """
    def _self_attention(self, layer, value, padding_mask, causal):
        length = value.size(1)
        causal_mask = torch.ones((length, length), dtype=torch.bool, device=value.device).triu(1) if causal else None
        if padding_mask is None:
            return layer(value, value, value, need_weights=False, attn_mask=causal_mask)[0]
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.dtype != torch.bool or padding_mask.shape != value.shape[:2]:
            raise ValueError('padding_mask must be a boolean tensor with shape (batch, sequence)')
        padding_mask = padding_mask.to(value.device)
        value = value.masked_fill(padding_mask.unsqueeze(-1), 0)
        mask = padding_mask[:, None, :].expand(-1, length, -1)
        if causal_mask is not None:
            mask = mask | causal_mask
        # Give otherwise fully blocked queries a zero-valued diagonal key.
        # This avoids NaNs in both native and fused attention; padded queries
        # are zeroed afterwards and contribute no gradients.
        blocked = mask.all(-1)
        diagonal = torch.eye(length, dtype=torch.bool, device=value.device)
        mask = mask & ~(blocked[:, :, None] & diagonal)
        output = layer(value, value, value, need_weights=False,
                       attn_mask=mask.repeat_interleave(layer.num_heads, dim=0))[0]
        return output.masked_fill(padding_mask.unsqueeze(-1), 0)
"""

DATA_STATE_HELPERS = """
    try:
        import numpy as np
    except ImportError:
        np = None

    def data_objects(loader):
        return {'dataset': getattr(loader, 'dataset', None),
                'sampler': getattr(loader, 'sampler', None),
                'batch_sampler': getattr(loader, 'batch_sampler', None)}

    def data_generators(loader):
        objects = {'loader': loader, **data_objects(loader)}
        return {name: obj.generator for name, obj in objects.items()
                if getattr(obj, 'generator', None) is not None}

    def validate_data_state():
        for name, loader in state_loaders.items():
            if getattr(loader, 'persistent_workers', False):
                raise ValueError(f'{name}: checkpoint/resume requires persistent_workers=False')
            for kind, obj in data_objects(loader).items():
                if hasattr(obj, 'state_dict') != hasattr(obj, 'load_state_dict'):
                    raise ValueError(f'{name} {kind} requires both state_dict and load_state_dict')
                if kind == 'dataset' and hasattr(obj, 'state_dict') and getattr(loader, 'num_workers', 0):
                    raise ValueError(f'{name}: stateful dataset checkpointing requires num_workers=0')

    def capture_data_state():
        loaders = {}
        for name, loader in state_loaders.items():
            if loader is None:
                continue
            loaders[name] = {
                'generators': {key: value.get_state() for key, value in data_generators(loader).items()},
                'objects': {key: {'class': type(obj).__module__ + '.' + type(obj).__qualname__,
                                  'state': obj.state_dict()}
                            for key, obj in data_objects(loader).items() if hasattr(obj, 'state_dict')},
            }
        numpy_state = None
        if np is not None:
            algorithm, keys, position, has_gauss, cached = np.random.get_state()
            numpy_state = (algorithm, keys.tolist(), position, has_gauss, cached)
        return {'loaders': loaders, 'numpy': numpy_state}

    def restore_data_state(saved):
        for name, entry in saved['loaders'].items():
            loader = state_loaders.get(name)
            if loader is None:
                raise ValueError(f'Checkpoint requires the {name} loader')
            generators = data_generators(loader)
            objects = data_objects(loader)
            for key in entry['generators']:
                if key not in generators:
                    raise ValueError(f'Checkpoint requires {name} {key} generator')
            for key, value in entry['objects'].items():
                obj = objects.get(key)
                if not hasattr(obj, 'load_state_dict') or type(obj).__module__ + '.' + type(obj).__qualname__ != value['class']:
                    raise ValueError(f'Checkpoint requires compatible {name} {key} state')
        if saved['numpy'] is not None and np is None:
            raise ValueError('Checkpoint requires NumPy for RNG restoration')
        for name, entry in saved['loaders'].items():
            loader = state_loaders[name]
            for key, value in entry['generators'].items():
                data_generators(loader)[key].set_state(value)
            for key, value in entry['objects'].items():
                data_objects(loader)[key].load_state_dict(value['state'])
        if saved['numpy'] is not None:
            algorithm, keys, position, has_gauss, cached = saved['numpy']
            np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
"""
