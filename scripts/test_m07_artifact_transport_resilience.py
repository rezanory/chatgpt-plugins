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
