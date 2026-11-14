from .edgequadd_model import EDGEQUADD
from .quadratic_components import ASPPQuadraticAdapter


def make_param_groups(model: EDGEQUADD, lr=1e-4, weight_decay=0.01):
    """One parameter group: every parameter of the model trains at `lr`."""
    return [{'params': list(model.parameters()), 'lr': lr, 'weight_decay': weight_decay}]


def split_quad_group(model: EDGEQUADD, groups, lr, weight_decay=0.0):
    """Move the quadratic branch into its own group at `lr`, leaving the rest of `groups` untouched.

    The branch is each adapter's low-rank projections (Ux/Vc/Wo, or the full `quad` Linear at rank 0), its scale alpha,
    and the ASPP that filters the correlation matrix feeding it. Used by `--quad_lr_mult` / `quad_lr_mult`.
    """
    adapters = [a for a in model.adapters if isinstance(a, ASPPQuadraticAdapter)]
    quad = {id(p)
            for a in adapters
            for m in ([a.Ux, a.Vc, a.Wo] if a.rank > 0 else [a.quad]) + [a.aspp]
            for p in m.parameters()}
    quad |= {id(a.alpha) for a in adapters}
    kept = [{**g, 'params': [p for p in g['params'] if id(p) not in quad]} for g in groups]
    return kept + [{'params': [p for p in model.parameters() if id(p) in quad],
                    'lr': lr, 'weight_decay': weight_decay}]
