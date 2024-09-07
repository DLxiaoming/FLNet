class Streamline_Fusion_module(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RGCSPELAN, self).__init__()
        
        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)
        
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        
        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)
        

    def forward_split(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))
