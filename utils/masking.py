import paddle


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with paddle.no_grad():
            self._mask = paddle.triu(paddle.ones(mask_shape, dtype=paddle.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        _mask = paddle.ones([L, scores.shape[-1]], dtype=paddle.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[paddle.arange(B)[:, None, None],
                             paddle.arange(H)[None, :, None],
                             index, :]
        self._mask = indicator.view(scores.shape)
    
    @property
    def mask(self):
        return self._mask
