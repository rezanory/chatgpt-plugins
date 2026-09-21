from pathlib import Path

W=Path(".github/workflows/pneumonia-v17-m07-continuation-20260914.yml")

def test_runtime_artifact_transport_is_non_blocking_but_identity_bound():
    t=W.read_text(encoding="utf-8")
    assert "runtime_kernel_ref: ${{ steps.runtime_identity.outputs.kernel_ref }}" in t
    assert "runtime_notebook_sha256: ${{ steps.runtime_identity.outputs.notebook_sha256 }}" in t
    assert "id: runtime_identity" in t
    assert "M07_EXPECTED_KERNEL_REF: ${{ needs.launch-m07-continuation.outputs.runtime_kernel_ref }}" in t
    assert "exported_kernel_ref=os.environ.get('M07_EXPECTED_KERNEL_REF','').strip()" in t
    for name in ("Preserve exact generated notebook","Preserve runtime orchestration receipt","Preserve runtime terminal evidence"):
        i=t.index("- name: "+name)
        block=t[i:i+260]
        assert "continue-on-error: true" in block, name
    i=t.index("- name: Download exact generated runtime artifact", t.index("verify-m07-runtime-terminal:"))
    assert "continue-on-error: true" in t[i:i+220]

def test_scientific_contract_still_precedes_submission():
    t=W.read_text(encoding="utf-8")
    assert t.index("- name: Validate dynamic M07 runtime artifact contract") < t.index("- name: Submit M07 continuation or approved Phase-2 kernel")
    assert "if($LASTEXITCODE -ne 0){throw 'M07 dynamic runtime artifact contract validation failed'}" in t


def test_phase2_recovery_verifies_existing_runs_without_launching_compute():
    text = W.read_text(encoding="utf-8")
    start = text.index("  recover-phase2-unit-terminal:")
    end = text.index("\n  verify-phase2-unit-terminal:", start)
    block = text[start:end]
    expected = {
        "M01": "35531382748",
        "M02": "35531386401",
        "M03": "35531394688",
        "M04": "35531397274",
        "M05": "35531402776",
        "M06": "35531405681",
        "M08": "35531409770",
        "M09": "35531415284",
        "M10": "35531419407",
        "M11": "35531422030",
        "M12": "35531424115",
    }
    assert "startsWith(github.event.inputs.recovery, 'PHASE2_VERIFY_')" in block
    assert "actions/download-artifact@v4" in block
    assert "phase2-unit-orchestration-receipt-${{ steps.recovery_identity.outputs.source_run_id }}" in block
    assert "scripts/pneumonia_phase2_runtime.py verify" in block
    assert "--provider-kernel-ref $env:SOURCE_PROVIDER_KERNEL_REF" in block
    assert "SaveKernel" not in block
    assert "PHASE2_INITIAL_LAUNCH_SHA: 2558d91f21c0ca1db5740d7c48d7cad4090f5dee" in block
    for model_id, run_id in expected.items():
        marker = f"PHASE2_VERIFY_{model_id}_R224_A01"
        token = f"PHASE2_UNIT_{model_id}_R224_A01"
        assert block.count(marker) == 1
        assert f"token='{token}';run_id='{run_id}'" in block
    expected_a03 = {
        "M01": "35539262689",
        "M02": "35539265230",
        "M03": "35539268161",
        "M04": "35539270533",
        "M05": "35539272554",
        "M06": "35539274778",
        "M08": "35539277136",
        "M09": "35539279566",
        "M10": "35539282287",
        "M11": "35539284809",
        "M12": "35539287827",
    }
    for model_id, run_id in expected_a03.items():
        marker = f"PHASE2_VERIFY_{model_id}_R224_A03"
        token = f"PHASE2_UNIT_{model_id}_R224_A03"
        assert block.count(marker) == 1
        assert f"token='{token}';run_id='{run_id}';source_sha='a52431acca1d3ae55f17c045f649d01a8b8a71c8'" in block
    expected_a09 = {
        "M01": "35565651205",
        "M04": "35565652015",
    }
    for model_id, run_id in expected_a09.items():
        marker = f"PHASE2_VERIFY_{model_id}_R224_A09"
        token = f"PHASE2_UNIT_{model_id}_R224_A09"
        assert block.count(marker) == 1
        expected_identity = (
            f"token='{token}';run_id='{run_id}';"
            "source_sha='eb17d2d6964864bba89b0d32fa17332dd3c30b78'"
        )
        assert expected_identity in block
    assert "-20[0-9]{6}-[0-9]+$" in block
    assert "source_sha=$sourceSha" in block
    assert "SOURCE_SHA: ${{ steps.recovery_identity.outputs.source_sha }}" in block


def test_phase2_launch_exports_exact_provider_kernel_ref_to_verifier():
    text = W.read_text(encoding="utf-8")
    assert "runtime_provider_kernel_ref: ${{ steps.runtime_submit.outputs.provider_kernel_ref }}" in text
    assert "id: runtime_submit" in text
    assert 'output.write(f"provider_kernel_ref={provider_kernel_ref}\\n")' in text
    assert "PHASE2_PROVIDER_KERNEL_REF: ${{ needs.launch-m07-continuation.outputs.runtime_provider_kernel_ref }}" in text
    assert "--provider-kernel-ref $env:PHASE2_PROVIDER_KERNEL_REF" in text
