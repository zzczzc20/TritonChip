#==============================================================================
# Package the RTL kernel into a .xo.
#
#   vivado -mode batch -source flow/fpga/package_xo.tcl -tclargs <repo> <out> [part] [variant]
#
# The variant is `none` (the on-chip build, also the default and an empty
# argument), `hbm` or `multicore-hbm`. It is written to <out>/tnca_variant.txt,
# which flow/fpga/link.sh reads.
#
# `hbm` (R40-04) packages rtl/board/tnca_hbm_xo.v with fpga/kernel_hbm.xml:
# the same kernel, name and register port, plus the AXI4 master m_axi_hbm.
#
# WHAT THIS PRODUCES, AND WHY IT IS SHAPED THIS WAY
#
# A .xo is a zip with three things in it:
#
#   <kernel>/kernel.xml        the copy of fpga/kernel.xml that travels with
#                              the kernel
#   ip_repo/<vlnv>/            a Vivado IP: component.xml, xgui/, src/, and the
#                              out-of-context constraints
#   xo.xml                     the manifest that names the two
#
# That is the unpacked shape of the RTL Kernel Wizard's own output
# (Vitis/2022.2/cli/examples/xo_kernel_srcs/rtl_kernel_wizard_0.xo), and it is
# the shape this reproduces -- because package_xo is documented as taking an
# -ip_directory and a -kernel_xml, and an IP is a thing that has to be built
# before it can be handed over.
#
# So this is two steps and not one. The first builds the IP from the same RTL
# the repository lints and simulates, and the second wraps it.
#
# WHY THERE IS NO AXI INFRARED HERE
#
# The AXI4-Lite interface is inferred from the port names, which is what the
# naming convention is for: `s_axi_tnca_awaddr`, `..._awvalid`, `..._wdata` and
# the rest are the names Vivado's inference recognises, and rtl/fpga/
# tnca_u50_top.v carries a comment saying why the platform's names are used
# rather than the adapter's own. If the inference finds nothing, the IP has no
# s_axi_tnca bus and package_xo fails on the kernel.xml referencing a port that
# is not there -- which is the failure this wants, announced, and not a kernel
# that links and then reads zero for ever.
#==============================================================================

set ROOT [file normalize [lindex $::argv 0]]
set OUT  [file normalize [lindex $::argv 1]]

# The card's part: the third argument, from flow/fpga/boards.sh's
# BOARD_PART; the U50's when absent, which is what this script always built.
set PART    [expr {[llength $::argv] > 2 ? [lindex $::argv 2] : "xcu50-fsvh2104-2-e"}]
set VARIANT [expr {[llength $::argv] > 3 ? [lindex $::argv 3] : "none"}]
if {$VARIANT eq ""} {set VARIANT none}
if {$VARIANT eq "hbm"} {
    set TOP     tnca_hbm_xo
    set KSRC    kernel_hbm.xml
    set BUSIFS  s_axi_tnca:m_axi_hbm
} elseif {$VARIANT eq "multicore-hbm"} {
    set TOP     tnca_multicore_hbm_xo
    set KSRC    kernel_multicore_hbm.xml
    set BUSIFS  s_axi_tnca:m_axi_hbm0:m_axi_hbm1:m_axi_hbm2
} elseif {$VARIANT eq "none"} {
    set TOP     tnca_u50_xo
    set KSRC    kernel.xml
    set BUSIFS  s_axi_tnca
} else {
    puts "TNCA_PKG_ERROR unknown variant '$VARIANT' (none, hbm, multicore-hbm)"
    exit 2
}
puts "TNCA_PKG_VARIANT=[expr {$VARIANT eq {none} ? {onchip} : $VARIANT}] TOP=$TOP"

set VENDOR  tnca.com
set LIB     kernel
set NAME    tnca_u50
set VERSION 1.0

# The kernel clock this design asks the platform for, and the one link.sh passes
# to v++ as --kernel_frequency 0:100 (it was 150 with the device divided; since
# R38's timing work the device runs on it undivided). The platform's default is
# 300; the header of flow/fpga/link.sh records why this design does not take it.
#
# Keeping this number in step with link.sh is not cosmetic: it is the frequency
# the IP declares, and a kernel whose metadata claims 300 while its link asks
# for 150 is a kernel with two answers to one question. flow/fpga/ooc.xdc
# constrains the same figure for out-of-context synthesis.
set AP_CLK_HZ 100000000

set IPDIR  $OUT/ip_repo/tnca_com_kernel_${NAME}_1_0
set KXML   $OUT/kernel.xml
set XO     $OUT/${NAME}.xo

file mkdir $OUT
file copy -force $ROOT/fpga/$KSRC $KXML

#------------------------------------------------------------------------------
# The RTL, named rather than globbed
#
# Named because a glob would take tmp_variants/ and anything else that happens
# to be sitting in the tree, and a kernel built from files nobody chose is a
# kernel nobody can account for. The lists mirror the Makefile's RTL_* so that
# a directory added there and forgotten here is visible as a difference.
#------------------------------------------------------------------------------
set RTL [list]
foreach d {common core gen nuca noc tile fpga} {
    foreach f [lsort [glob -nocomplain $ROOT/rtl/$d/*.v]] {
        lappend RTL $f
    }
}
foreach f [lsort [glob -nocomplain $ROOT/rtl/*.v]] {
    lappend RTL $f
}
lappend RTL $ROOT/rtl/board/$TOP.v

set INCS [list $ROOT/rtl/common $ROOT/rtl/noc]
set HEADERS [list]
foreach d $INCS {
    foreach f [lsort [glob -nocomplain $d/*.vh]] {lappend HEADERS $f}
}

puts "TNCA_PKG_ROOT=$ROOT"
puts "TNCA_PKG_FILES=[llength $RTL]"
puts "TNCA_PKG_HEADERS=[llength $HEADERS]"

#------------------------------------------------------------------------------
# The IP
#------------------------------------------------------------------------------
create_project -force tnca_xo $OUT/proj -part $PART

add_files -norecurse $RTL
add_files -norecurse $HEADERS
# Function templates are included separately inside each module.
set_property IS_GLOBAL_INCLUDE false [get_files $HEADERS]
# The mesh/dispatcher and elastic links use SystemVerilog generate syntax.
set_property FILE_TYPE SystemVerilog [get_files $RTL]
foreach d $INCS {
    set_property include_dirs $INCS [current_fileset]
}
add_files -fileset constrs_1 -norecurse $ROOT/flow/fpga/ooc.xdc
# The platform supplies the clock during implementation. Shipping this
# primary create_clock into the linked design cuts off the platform clock's
# source latency at the CU pin: HBM -> CU then appears to cross two clocks
# with about 5 ns of artificial skew (u280_r40a routed report).
set_property USED_IN {synthesis out_of_context} [get_files $ROOT/flow/fpga/ooc.xdc]

set_property top $TOP [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $IPDIR -vendor $VENDOR -library $LIB \
    -taxonomy /KernelIP -import_files

set core [ipx::current_core]
set_property name $NAME $core
set_property version $VERSION $core

#------------------------------------------------------------------------------
# The clock, associated with the bus it clocks
#
# Inference has already produced both interfaces by this point -- the log says
# `Inferred bus interface 's_axi_tnca'` and `'ap_clk'` -- and it does it off the
# port names alone, which is what the naming convention in rtl/fpga/
# tnca_u50_top.v is for. What it does NOT do is say that ap_clk is the clock for
# s_axi_tnca, and the log warns about that gap from each end:
#
#   WARNING: [IP_Flow 19-3158] Bus Interface 's_axi_tnca': FREQ_HZ bus parameter
#            missing from AXI interface when interface is not associated to a clock
#   WARNING: [IP_Flow 19-5661] Bus Interface 'ap_clk' does not have any bus
#            interfaces associated with it
#
# One association and one frequency close both, and they belong on the CLOCK
# interface, not on the AXI one. That is not a preference -- it is what the
# Wizard's own component.xml has, and matching that file is what this whole
# script is doing. From the shipped example, unpacked:
#
#   <busInterface ap_clk>
#     <parameter ASSOCIATED_BUSIF = "m00_axi:s_axi_control"/>
#     <parameter FREQ_HZ  resolve="user" = "250000000"/>
#
# Its AXI interface has no <parameters> block at all. With a clock behind it, it
# does not need one: FREQ_HZ is read off the associated clock rather than
# declared twice.
#
# WHY THIS IS NOT `set_property CONFIG.<name>`
#
# Because a bus parameter is not a component parameter. `CONFIG.ASSOCIATED_BUSIF`
# on a bus_interface object is an unknown property, which is the error that put
# this comment here; the parameter has to be fetched or created as a bus
# parameter object first.
#
# WHY A FAILURE BELOW IS REPORTED RATHER THAN FATAL
#
# Because what is at stake is metadata and not the kernel. The AXI ports exist in
# the IP either way and the platform connects them by name out of kernel.xml, so
# a missing association makes this IP carry less than the Wizard's does -- worth
# fixing, not worth failing a build over. Each step therefore says what happened
# and the file-level readback at the end is the evidence, rather than the absence
# of an error being the evidence. Note that the two warnings above are emitted
# during package_project, which has already run by the time this can fix
# anything, so their being in the log is expected and is not a leftover.
#------------------------------------------------------------------------------
set clk_if [ipx::get_bus_interfaces ap_clk -of_objects $core]

proc tnca_busparam {busif name value {resolve ""}} {
    set p [ipx::get_bus_parameters $name -of_objects $busif]
    if {[llength $p] == 0} {
        set p [ipx::add_bus_parameter $name $busif]
        puts "TNCA_PKG_PARAM $name: created"
    } else {
        puts "TNCA_PKG_PARAM $name: already present"
    }
    if {[catch {set_property value $value $p} why]} {
        puts "TNCA_PKG_PARAM $name: value refused -- $why"
        return
    }
    puts "TNCA_PKG_PARAM $name: value = [get_property value $p]"
    if {$resolve ne ""} {
        if {[catch {set_property value_resolve_type $resolve $p} why]} {
            puts "TNCA_PKG_PARAM $name: resolve type refused -- $why"
        } else {
            puts "TNCA_PKG_PARAM $name: resolve = [get_property value_resolve_type $p]"
        }
    }
}

# FREQ_HZ resolves as `user` for the same reason the Wizard's does: the number
# here is an estimate of what the platform will hand over, and a link that
# retargets the clock has to be able to say so. AP_CLK_HZ above is the estimate.
tnca_busparam $clk_if ASSOCIATED_BUSIF $BUSIFS
if {$VARIANT eq "multicore-hbm"} {
    tnca_busparam $clk_if ASSOCIATED_RESET ap_rst_n
    set reset_if [ipx::get_bus_interfaces ap_rst_n -of_objects $core]
    if {[llength $reset_if] != 1} {error "multicore HBM reset interface missing"}
    tnca_busparam $reset_if POLARITY ACTIVE_LOW
    foreach bus {s_axi_tnca m_axi_hbm0 m_axi_hbm1 m_axi_hbm2} {
        if {[llength [ipx::get_bus_interfaces $bus -of_objects $core]] != 1} {
            error "multicore HBM interface $bus missing"
        }
    }
}
tnca_busparam $clk_if FREQ_HZ $AP_CLK_HZ user

# A device profile other than the RTL defaults: TNCA_IMAGE_CONFIG names a Tcl
# file that sets TNCA_IMAGE_PARAMS, a dict of the wrapper's parameters and
# their values (written by the FPGA harness, or by R40's
# tools/r40/image_profile.py). Each name must be a parameter the packaged top
# exposes, or packaging stops: a profile is never built at a default it did
# not ask for. Without it the image is the defaults.
if {[info exists ::env(TNCA_IMAGE_CONFIG)]} {
    source $::env(TNCA_IMAGE_CONFIG)
    dict for {name value} $TNCA_IMAGE_PARAMS {
        set up [ipx::get_user_parameters $name -of_objects $core]
        set mp [ipx::get_hdl_parameters $name -of_objects $core]
        if {[llength $up] != 1 || [llength $mp] != 1} {
            error "TNCA_IMAGE_CONFIG: parameter $name is not exposed by $TOP"
        }
        set_property value $value $up
        set_property value $value $mp
        puts "TNCA_IMAGE_PARAM $name=$value"
    }
}

ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::save_core $core

puts "TNCA_PKG_IP=$IPDIR"

#------------------------------------------------------------------------------
# Read the association back out of the file that will ship
#
# Not out of the in-memory core, which is what just accepted the write and would
# agree with itself. component.xml is what package_xo zips.
#------------------------------------------------------------------------------
set fh [open $IPDIR/component.xml r]
set xml [read $fh]
close $fh

foreach {tag id} {ASSOC ASSOCIATED_BUSIF FREQ FREQ_HZ} {
    if {[regexp "BUSIFPARAM_VALUE\\.AP_CLK\\.$id\">(\[^<\]*)" $xml -> v]} {
        puts "TNCA_PKG_XML_$tag=$v"
    } else {
        puts "TNCA_PKG_XML_$tag=NOT IN component.xml"
    }
}

#------------------------------------------------------------------------------
# The container
#------------------------------------------------------------------------------
package_xo -kernel_xml $KXML -kernel_name $NAME \
    -ip_directory $IPDIR -xo_path $XO -force

set variant_file [open $OUT/tnca_variant.txt w]
puts $variant_file $VARIANT
close $variant_file
puts "TNCA_PKG_DONE=$XO"
