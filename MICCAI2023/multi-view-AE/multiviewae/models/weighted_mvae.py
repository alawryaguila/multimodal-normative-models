import torch
import hydra

from ..base.constants import MODEL_WEIGHTED_MVAE
from ..base.base_model import BaseModelVAE
from ..base.representations import weightedProductOfExperts

class weighted_mVAE(BaseModelVAE):
    """
    Multi-view Variational Autoencoder model with generalised Product of Experts (PoE) joint representation.

    Option to impose sparsity on the latent representations using a Sparse Multi-Channel Variational Autoencoder (http://proceedings.mlr.press/v97/antelmi19a.html)

    """

    def __init__(
        self,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):
        super().__init__(model_name=MODEL_WEIGHTED_MVAE,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.join_z = weightedProductOfExperts()
        tmp_weight = torch.FloatTensor(len(input_dim)-1, self.z_dim).fill_(1/len(input_dim))
        self.poe_weight = torch.nn.Parameter(data=tmp_weight, requires_grad=True)

    def encode(self, x):
        mu = []
        logvar = []
        for i in range(self.n_views):
            mu_, logvar_ = self.encoders[i](x[i])
            mu.append(mu_)
            logvar.append(logvar_)
        mu = torch.stack(mu)
        logvar = torch.stack(logvar)
        mu_out, logvar_out = self.join_z(mu, logvar, self.poe_weight)
        qz_x = hydra.utils.instantiate(
            self.cfg.encoder.default.enc_dist, loc=mu_out, scale=logvar_out.exp().pow(0.5)
        )
        with torch.no_grad():
            self.poe_weight = self.poe_weight.clamp_(0, +1)
        return [qz_x]

    def decode(self, qz_x):
        px_zs = []
        for i in range(self.n_views):
            px_z = self.decoders[i](qz_x[0]._sample(training=self._training))
            px_zs.append(px_z)
        return [px_zs]

    def forward(self, x):
        qz_x = self.encode(x)
        px_zs = self.decode(qz_x)
        fwd_rtn = {"px_zs": px_zs, "qz_x": qz_x}
        return fwd_rtn

    def calc_kl(self, qz_x):
        """
        VAE: Implementation from: https://arxiv.org/abs/1312.6114
        sparse-VAE: Implementation from: https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
        """
        if self.sparse:
            kl = qz_x[0].sparse_kl_divergence().sum(1, keepdims=True).mean(0)
        else:
            kl = qz_x[0].kl_divergence(self.prior).sum(1, keepdims=True).mean(0)
        return self.beta * kl

    def calc_ll(self, x, px_zs):
        ll = 0
        for i in range(self.n_views):
            ll += px_zs[0][i].log_likelihood(x[i]).sum(1, keepdims=True).mean(0)
        return ll

    def loss_function(self, x, fwd_rtn):
        px_zs = fwd_rtn["px_zs"]
        qz_x = fwd_rtn["qz_x"]

        kl = self.calc_kl(qz_x)
        ll = self.calc_ll(x, px_zs)

        total = kl - ll
        losses = {"loss": total, "kl": kl, "ll": ll}
        return losses

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()), 
                lr=self.learning_rate,
            )
            for i in range(self.n_views)
        ]
        optimizers.append(torch.optim.Adam([self.poe_weight],
                lr=self.learning_rate
            ))
        return optimizers