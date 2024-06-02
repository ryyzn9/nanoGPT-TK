import torch
import torch.nn as nn

def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len, dtype=torch.float), inv_freq).to(x)
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

def _rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def duplicate_interleave(m):
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def apply_rotary_pos_emb(x, sin, cos, scale=1):
    print("shape___2")
    print("x:", x.shape)
    print("sin before duplication:", sin.shape)
    print("cos before duplication:", cos.shape)

    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))

    print("sin after duplication:", sin.shape)
    print("cos after duplication:", cos.shape)

    # Adjusting dimensions to match x
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.unsqueeze(0).unsqueeze(2)

    print("sin adjusted:", sin.shape)
    print("cos adjusted:", cos.shape)

    return (x * cos[:, :, :x.shape[2], :x.shape[-1]]) + (_rotate_every_two(x) * sin[:, :, :x.shape[2], :x.shape[-1]])

class XPOS2(nn.Module):
    def __init__(self, head_dim, scale_base=512):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = 0
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]

        print("size of scale")
        print(scale.shape)
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        print("rpos shape")
        print(x.shape)
        return x

    def forward_reverse(self, x, offset=0, downscale=False):
        length = x.shape[1]
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        scale = self.scale ** torch.arange(min_pos, max_pos, 1).to(self.scale).div(self.scale_base)[:, None]
        sin, cos = fixed_pos_embedding(scale)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, -sin, cos, scale)
        return x

# Test
if __name__ == "__main__":
    x = torch.rand(1, 4, 6, 512).transpose(-1, -2)
    print("_____shape__ 3")
    print(x.shape)
    xpos = XPOS2(6)
    x_rot = xpos(x)
    # apply reverse
    x_rot_rev = xpos.forward(x)
    x_rot_ = xpos.forward_reverse(x)
    print(x_rot @ x_rot_rev.transpose(-1, -2))
    print(x_rot @ x_rot_.transpose(-1,-2))
    # print(x_rot)
