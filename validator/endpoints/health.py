from fastapi import APIRouter
from fastapi import status


router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def get_health():
    return {"status": "healthy"}


@router.post("/health", status_code=status.HTTP_200_OK)
async def post_health():
    return {"status": "healthy"}


def factory_router():
    return router
