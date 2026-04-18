import torch.nn as nn

from pointcept.models.modules import PointModule, PointSequential
from pointcept.utils.condition import canonicalize_condition, resolve_condition
from pointcept.models.builder import MODULES


@MODULES.register_module()
class CIM(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ALS", "ULS", "MLS"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = tuple(canonicalize_condition(cond) for cond in conditions)
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        condition = resolve_condition(
            point.condition,
            error_prefix="platform condition",
        )
        point.condition = condition
        if self.decouple:
            if condition not in self.conditions:
                raise ValueError(
                    f"Condition '{condition}' is not registered in CIM conditions {self.conditions}."
                )
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point
