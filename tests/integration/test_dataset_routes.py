async def test_create_and_get_dataset(client):
    create_response = await client.post(
        "/api/v1/datasets", json={"name": "test-dataset", "description": "desc"}
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["name"] == "test-dataset"

    get_response = await client.get(f"/api/v1/datasets/{created['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "test-dataset"


async def test_list_datasets(client):
    await client.post("/api/v1/datasets", json={"name": "a"})
    await client.post("/api/v1/datasets", json={"name": "b"})

    response = await client.get("/api/v1/datasets")

    assert response.status_code == 200
    names = [d["name"] for d in response.json()]
    assert "a" in names
    assert "b" in names


async def test_get_dataset_not_found(client):
    response = await client.get("/api/v1/datasets/999999")
    assert response.status_code == 404


async def test_delete_dataset(client):
    create_response = await client.post("/api/v1/datasets", json={"name": "to-delete"})
    dataset_id = create_response.json()["id"]

    delete_response = await client.delete(f"/api/v1/datasets/{dataset_id}")
    assert delete_response.status_code == 204

    get_response = await client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_response.status_code == 404