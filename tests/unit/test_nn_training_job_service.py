from app.services import nn_training_job_service
import torch
from app.services.nn_training_job_service import ChurnNet


class FakeJob:
    def __init__(self, status):
        self.id = 1
        self.dataset_id = 1
        self.status = status


async def test_run_training_job_skips_already_completed_job(mocker):
    fake_job = FakeJob(status="completed")
    mocker.patch("app.repositories.nn_training_job_repository.get_by_id", return_value=fake_job)
    update_mock = mocker.patch("app.repositories.nn_training_job_repository.update_status")

    await nn_training_job_service.run_training_job(job_id=1)

    update_mock.assert_not_called()


async def test_run_training_job_does_nothing_when_job_missing(mocker):
    mocker.patch("app.repositories.nn_training_job_repository.get_by_id", return_value=None)
    update_mock = mocker.patch("app.repositories.nn_training_job_repository.update_status")

    await nn_training_job_service.run_training_job(job_id=999)

    update_mock.assert_not_called()
    
    
def test_load_checkpoint_and_run_inference(tmp_path):
    model = ChurnNet(input_dim=45)
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    loaded_model = ChurnNet(input_dim=45)
    loaded_model.load_state_dict(torch.load(checkpoint_path))
    loaded_model.eval()

    sample_input = torch.zeros(1, 45)
    with torch.no_grad():
        output = loaded_model(sample_input)

    assert output.shape == (1, 1)