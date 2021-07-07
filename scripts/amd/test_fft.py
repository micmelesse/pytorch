from torch.cuda import device
import torch


def gen_like_montonic_tensor(tensor):
    line = torch.arange(0, tensor.numel(), device="cuda")
    return torch.reshape(line, tensor.shape)


def test_gen_tensor():
    montonic_tensor = gen_like_montonic_tensor(torch.rand(8, 7))
    print(montonic_tensor)
    return montonic_tensor


def zero_last_col(b):
    real_b = b.real
    real_b = real_b.type(torch.cfloat)
    new_b = torch.cat([b[:, :-1], real_b[:, -1:]], axis=1)  # shape=(8, 4)
    return new_b


def test_rfft2():
    H, W = 8, 7

    a = gen_like_montonic_tensor(torch.rand(H, W))
    print("a", a.shape, a)

    b = torch.fft.rfft2(a)  # shape=(8, 4)
    print("b", b.shape, b)
    b_2 = zero_last_col(b)
    print("b_2", b_2.shape, b_2)

    c = torch.fft.irfft2(b)  # shape=(8, 6)
    print("c", c.shape, c)


def test_rfft2_2():
    H, W = 4, 5

    a = gen_like_montonic_tensor(torch.rand(H, W))
    print("a", a.shape, a)

    b = torch.fft.rfft2(a)  # shape=(8, 4)
    print("b", b.shape, b)
    b_2 = zero_last_col(b)
    print("b_2", b_2.shape, b_2)

    c = torch.fft.irfft2(b,dim = (1, 0))  # shape=(8, 6)
    print("c", c.shape, c)

# test_hfft()
# test_gen_tensor()

test_rfft2_2()
