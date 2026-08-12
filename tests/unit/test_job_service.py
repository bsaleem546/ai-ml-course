from app.schemas.dataset import JobStatus
from app.services import job_service


class FakeJob:
    def __init__(self, status):
        self.id = 1
        self.dataset_id = 1
        self.status = status


async def test_run_ingestion_job_skips_already_completed_job(mocker):
    fake_job = FakeJob(status=JobStatus.COMPLETED)
    mocker.patch("app.repositories.job_repository.get_by_id", return_value=fake_job)
    update_mock = mocker.patch("app.repositories.job_repository.update_status")

    await job_service.run_ingestion_job(job_id=1)

    update_mock.assert_not_called()


async def test_run_ingestion_job_does_nothing_when_job_missing(mocker):
    mocker.patch("app.repositories.job_repository.get_by_id", return_value=None)
    update_mock = mocker.patch("app.repositories.job_repository.update_status")

    await job_service.run_ingestion_job(job_id=999)

    update_mock.assert_not_called()