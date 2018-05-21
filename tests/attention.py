import torch
import numpy as np

torch.manual_seed(2)

from nmtpytorch.layers import Attention, Attentionv2

SSEQ = 55
TSEQ = 60
BS = 64
CTX = 512
HID = 256


hid = torch.autograd.Variable(torch.rand(TSEQ, BS, HID))
ctx = torch.autograd.Variable(torch.rand(SSEQ, BS, CTX))
all_ones = torch.autograd.Variable(torch.ones(SSEQ, BS))
minus1 = all_ones.clone()
minus1[SSEQ-1, :] = 0


def test_att(method='mlp', mask=None):
    att2 = Attentionv2(CTX, HID, CTX, method=method)
    att = Attention(CTX, HID, 'ctx', att_type=method)

    if method == 'mlp':
        att.mlp.weight.set_(att2.ff.weight)
    att.ctx2ctx.weight.set_(att2.ctx2mid.weight)
    att.hid2ctx.weight.set_(att2.hid2mid.weight)
    att.ctx2hid.weight.set_(att2.att2hid.weight)

    alpha2, vec2 = att2(hid, ctx, mask)
    print(vec2.shape, alpha2.shape)

    alpha, vec = [], []
    alpha2_t, vec2_t = [], []
    for i in range(TSEQ):
        alpha_i, vec_i = att(hid[i].unsqueeze(0), ctx, mask)
        alpha2_i, vec2_i = att2(hid[i].unsqueeze(0), ctx, mask)
        alpha.append(alpha_i)
        vec.append(vec_i)

        alpha2_t.append(alpha2_i.squeeze(0))
        vec2_t.append(vec2_i.squeeze(0))

    vec = torch.stack(vec)
    vec2_t = torch.stack(vec2_t)
    alpha = torch.stack(alpha) # --> T, S, B
    alpha2_t = torch.stack(alpha2_t)

    assert np.allclose(vec.data, vec2.data, rtol=0, atol=1e-3), "FAIL"
    assert np.allclose(alpha.data, alpha2.data, rtol=0, atol=1e-3), "FAIL"
    assert np.allclose(alpha2_t.data, alpha2.data, rtol=0, atol=1e-3), "FAIL"
    assert np.allclose(vec2_t.data, vec2.data, rtol=0, atol=1e-3), "FAIL"

if __name__ == '__main__':
    test_att('mlp')
    test_att('dot')
    test_att('mlp', mask=all_ones)
    test_att('mlp', mask=minus1)

