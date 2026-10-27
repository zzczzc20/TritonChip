# Coherent logical layout across core outlets and host DMA

## Implemented layout

The optional `HBM_STRIPED=1` image layout distributes logical 4 KiB pages round-robin over all 32 pseudo-channels. For BO-relative logical address L, bits [16:12] select the PC, bits [32:17] become the high part of that PC's local offset, and bits [11:0] remain the byte offset within a page. The physical address is `{L[16:12], L[32:17], L[11:0]}`. This is a bijection over the complete 8 GiB allocation.

`tnca_line_axi` retains logical addresses as the same-line ordering/hazard key and changes only the outgoing AXI address. Its striped aperture accepts only physical base zero and exactly 8 GiB. The option passes through the multicore shell and board wrapper and defaults to zero. Legacy image parameters and linear addressing remain unchanged.

The multicore control binding advertises capability bit 64 at register 0xb8 and the read-only layout signature `0x5234315300010c05` at 0x120. Software probes the signature only after the capability bit. This avoids aliasing extended registers in older bindings.

`python/tnca/hbm_layout.py` provides the same permutation and inverse, validates transfer bounds and coalesces the pages for each PC into one contiguous DMA request. A contiguous upload or download therefore needs at most 32 requests, rather than one per 4 KiB page. Scatter/gather restores the caller's logical byte order, including partial first and last pages.

The multicore XRT link enables this layout only after validating the signature, an explicitly requested 8 GiB BO, physical BO base zero and image metadata proving 32 used, complete 256 MiB banks tagged PC0 through PC31 at the expected physical addresses. Missing, reordered or malformed physical geometry refuses. After an allocation refusal, DMA cannot silently fall back to linear addressing. The backend also checks that DMA is using a layout-aware link before admitting a striped window.

## Verification

`python3 tools/r41/hbm_layout_tests.py` passes nine tests, including:

- 2,006 logical addresses checked against an independent page/PC division oracle, with inverse round trips.
- Sparse DMA transfers across page, PC, 4 GiB and end-of-allocation boundaries; each transfer coalesces into at most 32 requests. Byte samples are checked against the independent physical oracle before the complete download is compared.
- Capability, allocation size/base, physical topology and post-refusal DMA guards. Legacy DMA remains linear and never reads the extended layout register.
- 100 real RTL line-adapter requests across all PCs and the last 8 GiB line. The AXI slave returns the actual AXI address as data, which must match the independent oracle.
- RTL refusal of nonzero-base and partial-size striped apertures, plus all four combinations of layout and runtime-memory-control capability. The layout signature is read-only.

The original line-adapter suite passes all 78 configurations and eight sustained stream cases with the default linear layout. Existing multicore option, topology and host memory-control tests pass. Raw logs and source fingerprints accompany this document.

An initial compatibility-test run failed because its allocation-only fixture had no register-read method. Capability selection now needs that method. The failure remains in `legacy_fixture_failure.log`; the fixture now explicitly returns a legacy capability and refuses any extended-register probe. Alias-group metadata is checked before the hardware capability read. The updated test retains its pre-allocation refusal checks and passes.

## Limits and next work

This checkpoint establishes host-DMA/RTL address agreement and capability refusal. It does not establish a new image, board bandwidth, a full-suite regression on the new layout, or the complete cache-ownership protocol around accelerator commands. Native storage/tensor copies still execute through the existing core path until stream control is integrated.

Next implement the replicated stream control/binding, attach the actual tensor-copy path with serialized cache handoff and verify the combination. Full-kernel/tile placement and crossing screens must precede a full implementation run. Physical BO placement and metadata checks must be exercised on the eventual approved image; fixture topology is functional evidence only.
