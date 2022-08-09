from sloop.models.nn.iter5.pr_to_fref_model import PrToFrefModel as Iter1PrToFrefModel
from sloop.models.nn.iter5.fref_to_pr_model import FrefToPrModel as Iter1FrefToPrModel
from sloop.models.nn.iter0.point_model import PointModel as Iter0PointModel
from sloop.models.nn.iter0.point_polar_model import PointPolarModel as Iter0PointPolarModel
from sloop.models.nn.iter0.point_absolute_model import PointAbsoluteModel as Iter0PointAbsoluteModel
from sloop.models.nn.iter0.point_absolute_polar_model import PointAbsolutePolarModel as Iter0PointAbsolutePolarModel
from sloop.models.nn.iter0.map_point_foref_model import MapPointForefModel as Iter0MapToForefModel
from sloop.models.nn.iter1.context_foref_model import ContextForefModel as Iter1ContextForefModel
from sloop.models.nn.iter1.bdg_foref_model import LandmarkForefModel as Iter1LandmarkForefModel
from sloop.models.nn.iter1.fref_to_pr_model import ContextPrModel as Iter1ContextPrModel
from sloop.models.nn.iter1.point_foref_model import PointForefModel as Iter1PointForefModel
from sloop.models.nn.iter2.ego_context_foref_model import EgoCtxForefAngleModel as Iter2EgoCtx
from sloop.models.nn.iter2.ego_bdg_foref_model import EgoBdgForefAngleModel as Iter2EgoBdg
from sloop.models.nn.iter2.context_foref_model import CtxForefAngleModel as Iter2Ctx
from sloop.models.nn.iter2.random_model import RandomModel as Iter2Random

MODELS_BY_ITER = {
    0: {Iter0PointModel.NAME: Iter0PointModel,
        Iter0PointPolarModel.NAME: Iter0PointPolarModel,
        Iter0PointAbsoluteModel.NAME: Iter0PointAbsoluteModel,
        Iter0PointAbsolutePolarModel.NAME: Iter0PointAbsolutePolarModel,
        Iter0MapToForefModel.NAME: Iter0MapToForefModel},

    1: {Iter1ContextForefModel.NAME: Iter1ContextForefModel,
        Iter1ContextPrModel.NAME: Iter1ContextPrModel,
        Iter1LandmarkForefModel.NAME: Iter1LandmarkForefModel,
        Iter1PointForefModel.NAME: Iter1PointForefModel},

    2: {Iter2Ctx.NAME: Iter2Ctx,
        Iter2EgoBdg.NAME: Iter2EgoBdg,
        Iter2EgoCtx.NAME: Iter2EgoCtx,
        Iter2Random.NAME: Iter2Random},

    5: {Iter1PrToFrefModel.NAME: Iter1PrToFrefModel,
        Iter1FrefToPrModel.NAME: Iter1FrefToPrModel}
}


from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
