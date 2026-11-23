#!/bin/bash
# build_report.sh -- evidence/r42/kfuse_rtl/REPORT.md = narrative parts + the generated tables
set -eu
EV=$(cd $(dirname $0)/../../.. && pwd)/evidence/r42/kfuse_rtl
cd $EV
{
    cat report_part_head.md; echo
    cat report_part_struct.md; echo
    cat report_part_numerics.md; echo
    cat report_part_verif.md; echo
    cat report_part_res.md; echo
    echo '## Tables'; echo
    cat tables.md; echo
    cat report_part_limits.md; echo
    cat report_part_files.md
} > REPORT.md
echo wrote $EV/REPORT.md
