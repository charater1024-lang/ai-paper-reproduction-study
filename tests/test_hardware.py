from llm_engineering_lab.hardware import HardwareProfile, recommend_training_profile


def test_hardware_profile_and_safe_defaults() -> None:
    hardware = HardwareProfile.detect()
    profile = recommend_training_profile(hardware)

    assert hardware.cpu_threads >= 1
    assert profile.device in {"cpu", "cuda"}
    assert profile.dataloader_workers == 0
    assert profile.transformer_width % 8 == 0
    assert profile.sequence_length <= 256
    if profile.device == "cuda":
        assert hardware.cuda_available
        assert profile.use_amp
