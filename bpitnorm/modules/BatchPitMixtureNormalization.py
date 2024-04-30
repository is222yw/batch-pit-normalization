from torch import nn



class BatchPitMixtureNorm1d(nn.Module):
    """
    Just a stub for now. However, in this class we will implement the PIT
    using a mixture (actually, a convex combination) of various CDFs. Each
    of these may have its own, learnable parameters. The goal of this class
    is then to learn some best weights for this convex combination.
    """
    pass