# OOC estimate of one AXI4 data-width converter 512 -> 256 (what v++ inserts in front of a 256-bit HBM port
# for a 512-bit kernel master); defaults as the linker would use them (no FIFO mode), 33-bit address, ID 6.
set_param general.maxThreads 4
create_project -in_memory -part xcu280-fsvh2892-2L-e
create_ip -name axi_dwidth_converter -vendor xilinx.com -library ip -module_name dwc
set_property -dict [list CONFIG.SI_DATA_WIDTH {512} CONFIG.MI_DATA_WIDTH {256} CONFIG.ADDR_WIDTH {33} CONFIG.SI_ID_WIDTH {6} CONFIG.PROTOCOL {AXI4}] [get_ips dwc]
generate_target all [get_ips dwc]
synth_ip [get_ips dwc]
set f [get_files dwc.xci]
synth_design -top dwc -part xcu280-fsvh2892-2L-e -mode out_of_context
create_clock -period 10 [get_ports s_axi_aclk]
opt_design
place_design
route_design
report_utilization -file dwc_util.rpt
puts "DWC_DONE"
