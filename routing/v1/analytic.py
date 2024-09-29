from fastapi import APIRouter, status, Depends

from schemas.analytic import ChatReqeust
from services.analytic import AnalyticService

router = APIRouter(prefix="/api/v1/analytic", tags=["analytic"])


@router.post(
    "/",
    summary="создание аналитки",
)
async def answer(
    req: ChatReqeust,
    analytic_service: AnalyticService = Depends(AnalyticService),
):
    await analytic_service.create(req)

    return status.HTTP_200_OK
