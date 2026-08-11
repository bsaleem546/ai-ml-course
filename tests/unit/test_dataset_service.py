import pytest

from app.services import dataset_service
from app.services.dataset_service import DatasetNotFoundError

class FakeDataset:
    def __init__(self, id, name, description=None):
        self.id = id
        self.name = name
        self.description = description


async def test_get_dataset_returns_dataset_when_found(mocker):
    fake = FakeDataset(id=1, name="test")
    mocker.patch(
        "app.repositories.dataset_repository.get_by_id",
        return_value=fake,
    )

    result = await dataset_service.get_dataset(db=None, dataset_id=1)

    assert result is fake


async def test_get_dataset_raises_when_not_found(mocker):
    mocker.patch(
        "app.repositories.dataset_repository.get_by_id",
        return_value=None,
    )

    with pytest.raises(DatasetNotFoundError):
        await dataset_service.get_dataset(db=None, dataset_id=999)


async def test_delete_dataset_calls_repository_delete(mocker):
    fake = FakeDataset(id=1, name="test")
    mocker.patch("app.repositories.dataset_repository.get_by_id", return_value=fake)
    delete_mock = mocker.patch("app.repositories.dataset_repository.delete")

    await dataset_service.delete_dataset(db=None, dataset_id=1)

    delete_mock.assert_called_once_with(None, fake)