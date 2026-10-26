# Diagnostic candidate: retain the three-core instruction implementation,
# enable the wide hybrid memory path and four parallel SFU lanes.
# Full-image and board acceptance remain outstanding.
source [file normalize [file join [file dirname [info script]] .. .. stage2 integration image.tcl]]
dict set TNCA_IMAGE_PARAMS HBM_STRIPED 1
dict set TNCA_IMAGE_PARAMS HYBRID_LSU 1
dict set TNCA_IMAGE_PARAMS SFU_LANES 4
