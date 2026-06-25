"""Behavior tests for CBZ archive batching and I/O orchestration."""

import os
import zipfile

import numpy as np
import pytest

from manga_translator.pipelines.cbz import CbzPipeline


TEST_IMAGE_SHAPE = (2, 2, 3)
TOTAL_ARCHIVE_IMAGES = 5
EXPECTED_ARCHIVE_FILENAMES = ["a/page1.png", "b/page2.png"]
EXPECTED_BATCH_CALLS = [2, 2, 1]


class FakeImageToImage:
    def __init__(self):
        self.calls = []

    async def __call__(self, images):
        self.calls.append(len(images))
        return images


@pytest.mark.asyncio
async def test_cbz_pipeline_rejects_non_positive_batch_size(tmp_path):
    """Batch size should be validated early to prevent invalid processing loops."""
    pipeline = CbzPipeline(FakeImageToImage())

    with pytest.raises(ValueError):
        await pipeline(str(tmp_path / "in.cbz"), str(tmp_path / "out"), batch_size=0)


def test_extract_zip_returns_sorted_non_directory_entries(tmp_path):
    """Zip extraction should ignore directory placeholders and return deterministic order."""
    archive = tmp_path / "sample.cbz"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("b/page2.png", b"x")
        zf.writestr("a/", b"")
        zf.writestr("a/page1.png", b"y")

    out_dir = tmp_path / "extract"
    out_dir.mkdir()

    filenames = CbzPipeline.extract_zip(str(archive), str(out_dir))

    assert filenames == EXPECTED_ARCHIVE_FILENAMES


@pytest.mark.asyncio
async def test_cbz_pipeline_processes_in_batches_and_writes_outputs(
    tmp_path, monkeypatch
):
    """Pipeline should process all images, including the tail batch, and write each output."""
    archive = tmp_path / "input.cbz"
    with zipfile.ZipFile(archive, "w") as zf:
        for i in range(TOTAL_ARCHIVE_IMAGES):
            zf.writestr(f"pages/{i}.png", b"not-real-image-bytes")

    writes = []

    async def fake_read_image(path):
        idx = int(os.path.basename(path).split(".")[0])
        return np.full(TEST_IMAGE_SHAPE, idx, dtype=np.uint8)

    async def fake_write_image(path, image):
        writes.append((path, int(image[0, 0, 0])))
        return True

    monkeypatch.setattr(CbzPipeline, "read_image", staticmethod(fake_read_image))
    monkeypatch.setattr(CbzPipeline, "write_image", staticmethod(fake_write_image))

    fake = FakeImageToImage()
    pipeline = CbzPipeline(fake)

    out_dir = tmp_path / "out"
    ok = await pipeline(str(archive), str(out_dir), batch_size=2)

    assert ok is True
    assert fake.calls == EXPECTED_BATCH_CALLS
    assert len(writes) == TOTAL_ARCHIVE_IMAGES
    written_values = sorted(value for _, value in writes)
    assert written_values == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_cbz_read_image_and_write_image_static_helpers(tmp_path):
    """CBZ helper methods should persist and read back image data via OpenCV wrappers."""
    image = np.full(TEST_IMAGE_SHAPE, 123, dtype=np.uint8)
    image_path = tmp_path / "img.png"

    await CbzPipeline.write_image(str(image_path), image)
    read_back = await CbzPipeline.read_image(str(image_path))

    assert read_back is not None
    assert read_back.shape == image.shape
    assert int(read_back[0, 0, 0]) == 123
