# Preserve the existing three-core x8 instruction implementation and enable
# only the explicit 32-PC logical memory layout for this image candidate.
source [file normalize [file join [file dirname [info script]] .. .. stage2 integration image.tcl]]
dict set TNCA_IMAGE_PARAMS HBM_STRIPED 1
