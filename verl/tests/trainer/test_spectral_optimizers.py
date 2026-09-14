import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from htmuon import htmuon_transform, SingleDeviceHTMuonWithAuxAdam
from soren_lamdba import soren_lamdba_transform, sqrt_coefficients


@pytest.mark.parametrize('shape', [(4, 13), (13, 4), (5, 5)])
@pytest.mark.parametrize('mode', ['htmuon', 'exact'])
def test_gram_matches_svd(shape, mode):
    torch.manual_seed(2)
    x = torch.randn(shape, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(x / x.norm(), full_matrices=False)
    if mode == 'htmuon':
        actual = htmuon_transform(x)
        f = s.pow(.125)
    else:
        actual = soren_lamdba_transform(x, .3, mode)
        root = (s.square() + .3).sqrt()
        denominator = 1 + 1.3**.5
        f = (s + root) / denominator
    torch.testing.assert_close(actual, (u * f) @ vh, atol=1e-10, rtol=1e-8)


def test_zero_and_rank_deficient():
    for mode in ['polynomial', 'exact']:
        assert torch.count_nonzero(soren_lamdba_transform(torch.zeros(3, 8), mode=mode)) == 0
    x = torch.zeros(3, 8)
    x[0, 0] = 1
    torch.testing.assert_close(htmuon_transform(x), x)


def test_polynomial_lambda_and_orientation():
    x = torch.randn(3, 12)
    for lamdba in [.01, .4, .8]:
        a = soren_lamdba_transform(x, lamdba)
        torch.testing.assert_close(a, soren_lamdba_transform(x.T, lamdba).T)
        assert torch.isfinite(a).all()
    assert sqrt_coefficients(.01) != sqrt_coefficients(.4)


def test_aux_adam_and_resume():
    main = torch.nn.Parameter(torch.randn(3, 8))
    aux = torch.nn.Parameter(torch.randn(4))
    ref = torch.nn.Parameter(aux.detach().clone())
    opt = SingleDeviceHTMuonWithAuxAdam([
        dict(params=[main], use_muon=True),
        dict(params=[aux], use_muon=False, lr=.001, weight_decay=.1),
    ])
    adam = torch.optim.AdamW([ref], lr=.001, betas=(.9, .95), eps=1e-8, weight_decay=.1)
    for _ in range(3):
        main.grad = torch.randn_like(main)
        aux.grad = torch.randn_like(aux)
        ref.grad = aux.grad.clone()
        opt.step()
        adam.step()
    torch.testing.assert_close(aux, ref)
    import copy
    clone_main = torch.nn.Parameter(main.detach().clone())
    clone_aux = torch.nn.Parameter(aux.detach().clone())
    restored = SingleDeviceHTMuonWithAuxAdam([
        dict(params=[clone_main], use_muon=True), dict(params=[clone_aux], use_muon=False)
    ])
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    clone_main.grad = main.grad.clone()
    clone_aux.grad = aux.grad.clone()
    opt.step()
    restored.step()
    torch.testing.assert_close(main, clone_main)
    torch.testing.assert_close(aux, clone_aux)
