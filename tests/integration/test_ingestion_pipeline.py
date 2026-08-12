import io

async def test_upload_creates_job_and_completes(client):
    csv_content = b"name,age\nAlice,30\nBob,25\n"
    response = await client.post(
        "/api/v1/datasets/upload",
        data={"name": "pipeline-test"},
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )

    assert response.status_code == 202
    job = response.json()
    assert job["status"] in ("queued", "running", "completed")

    job_response = await client.get(f"/api/v1/jobs/{job['id']}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "completed"


async def test_upload_non_csv_rejected(client):
    response = await client.post(
        "/api/v1/datasets/upload",
        data={"name": "bad-upload"},
        files={"file": ("test.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert response.status_code == 400


async def test_get_nonexistent_job_returns_404(client):
    response = await client.get("/api/v1/jobs/999999")
    assert response.status_code == 404